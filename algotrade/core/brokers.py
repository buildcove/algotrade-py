import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import ib_async
import pandas as pd
import pytz
from algotrade.schedules import IB_TZ
from structlog import get_logger

from . import types, utils
from .repository import Repository

logger = get_logger()


class BaseBroker:
    def __init__(self, repository: Repository):
        self.repository: Repository = repository
        self._order_types = {
            types.LimitOrder: self._limit_order,
            types.MarketOrder: self._market_order,
            types.StopOrder: self._stop_order,
            types.BracketOrder: self._bracket_order,
        }

    def order(self, order: types.BaseOrder):
        instance_class = order.__class__
        if instance_class in self._order_types:
            return self._order_types[instance_class](order)
        else:
            raise TypeError(f"Unknown order type: {order}")

    def _limit_order(self, order: types.LimitOrder):
        raise NotImplementedError

    def _market_order(self, order: types.MarketOrder):
        raise NotImplementedError

    def _stop_order(self, order: types.StopOrder):
        raise NotImplementedError

    def _bracket_order(self, order: types.BracketOrder):
        # Each orders are separated.
        raise NotImplementedError

    def get_orders(self, contract):
        raise NotImplementedError

    def cancel_orders(self, contract):
        raise NotImplementedError


class BacktestingBroker(BaseBroker):
    def __init__(self, repository: Repository, tick_size: float = 1 / 64, limit_spread_ticks: int = 1):
        super().__init__(repository)
        self.orders: List[types.BaseOrder] = []
        self.positions: List[types.Position] = []
        self.closed_positions: List[types.ClosedPosition] = []

        self.tick_size = tick_size
        self.limit_spread_ticks = limit_spread_ticks
        # Track bracket child orders that should only become active once an entry
        # position exists (prevents TP/SL from triggering before entry fills).
        self._bracket_child_requires: dict[int, types.Action] = {}

    def _limit_order(self, order: types.LimitOrder):
        self.orders.append(order)

    def _market_order(self, order: types.MarketOrder):
        self.orders.append(order)

    def _stop_order(self, order: types.StopOrder):
        self.orders.append(order)

    def _bracket_order(self, order: types.BracketOrder):
        # Model brackets as independent child orders so the backtesting executor can
        # fill entry/TP/SL based on the stream.
        self.orders.append(order.order)
        if order.take_profit is not None:
            self.orders.append(order.take_profit)
            self._bracket_child_requires[id(order.take_profit)] = order.order.action
        if order.stop_loss is not None:
            self.orders.append(order.stop_loss)
            self._bracket_child_requires[id(order.stop_loss)] = order.order.action

    def next(self, stream: types.Stream):
        self.executor(stream)
        self.close_positions()

    def executor(self, stream: types.Stream):
        for order in list(self.orders):
            required = self._bracket_child_requires.get(id(order))
            if required is not None:
                has_required_position = any(
                    position.symbol == order.symbol and position.action == required for position in self.positions
                )
                if not has_required_position:
                    continue
            execute = False
            entry_price = 0
            bracket = None

            if isinstance(order, types.BracketOrder):
                bracket = order
                order = bracket.order

            if isinstance(order, types.MarketOrder):
                execute = True
                entry_price = stream.ask if order.action == types.Action.LONG else stream.bid

            elif isinstance(order, types.LimitOrder):
                # this isn't an actual limit order, but you won't get filled since our data is 1 min.
                # so we're just going to simulate it with ticks spread
                if order.action == types.Action.LONG and order.price <= stream.ask:
                    execute = True
                    entry_price = order.price

                elif order.action == types.Action.SHORT and stream.bid >= order.price:
                    execute = True
                    entry_price = order.price

            elif isinstance(order, types.StopOrder):
                if order.action == types.Action.LONG and stream.bid >= order.price:
                    execute = True
                    entry_price = stream.bid
                elif order.action == types.Action.SHORT and stream.ask <= order.price:
                    execute = True
                    entry_price = stream.ask

            if execute:
                position = types.Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    action=order.action,
                    entry_price=entry_price,
                    entry_timestamp=stream.timestamp,
                )
                self.positions.append(position)
                logger.debug("Order filled", order=order, position=position)

                if isinstance(bracket, types.BracketOrder):
                    if bracket.take_profit is not None:
                        logger.debug("Take profit order placed", order=bracket.take_profit)
                        self.orders.append(bracket.take_profit)
                    if bracket.stop_loss is not None:
                        logger.debug("Stop loss order placed", order=bracket.stop_loss)
                        self.orders.append(bracket.stop_loss)

                    order = bracket

                self.repository.insert_position(order)
                if order in self.orders:
                    self.orders.remove(order)
                self._bracket_child_requires.pop(id(order), None)

    def close_positions(self):
        group_by_symbol = {}
        for position in self.positions:
            if position.symbol not in group_by_symbol:
                group_by_symbol[position.symbol] = []
            group_by_symbol[position.symbol].append(position)

        for symbol, positions in group_by_symbol.items():
            long_quantity = 0
            short_quantity = 0

            long_positions = []
            short_positions = []

            for position in positions:
                if position.action == types.Action.LONG:
                    long_quantity += position.quantity
                    long_positions.append(position)
                elif position.action == types.Action.SHORT:
                    short_quantity += position.quantity
                    short_positions.append(position)

            if long_quantity != short_quantity:
                continue

            long_entry_count = len(long_positions)
            short_entry_count = len(short_positions)

            total_long_prices = sum(position.entry_price for position in long_positions)
            total_short_prices = sum(position.entry_price for position in short_positions)

            action = types.Action.LONG if positions[-1].action == types.Action.SHORT else types.Action.SHORT

            average_entry_price = (
                total_long_prices / long_entry_count if action == types.Action.LONG else total_short_prices / short_entry_count
            )
            average_exit_price = (
                total_short_prices / short_entry_count if action == types.Action.LONG else total_long_prices / long_entry_count
            )
            if action == types.Action.LONG:
                gain = (average_exit_price - average_entry_price) / self.tick_size
            else:
                gain = (average_entry_price - average_exit_price) / self.tick_size

            position = types.ClosedPosition(
                symbol=symbol,
                quantity=long_quantity,
                action=action,
                entry_price=average_entry_price,
                entry_timestamp=positions[0].entry_timestamp,
                exit_price=average_exit_price,
                exit_timestamp=positions[-1].entry_timestamp,
                gain_ticks=gain,
            )
            self.closed_positions.append(position)
            logger.debug("Position closed", position=asdict(position))

            # remove closed positions
            self.positions = [position for position in self.positions if position.symbol != symbol]
            # cancel any remaining orders for the symbol (e.g., orphaned TP/SL)
            remaining = []
            for order in self.orders:
                if order.symbol == symbol:
                    self._bracket_child_requires.pop(id(order), None)
                    continue
                remaining.append(order)
            self.orders = remaining

    def get_orders(self, contract):
        return [order for order in self.orders if order.symbol == contract]

    def get_positions(self, contract):
        return [position for position in self.positions if position.symbol == contract]

    def close_position(self, contract):
        logger.debug("Closing position", symbol=contract)
        positions = self.get_positions(contract)
        for position in positions:
            action = types.Action.SHORT if position.action == types.Action.LONG else types.Action.LONG
            order = types.MarketOrder(
                symbol=contract,
                quantity=position.quantity,
                action=action,
                timestamp=utils.get_utc(),
            )
            self._market_order(order)

        # remove closed positions
        self.positions = [position for position in self.positions if position.symbol != contract]

    def cancel_orders(self, orders: List[types.BaseOrder]):
        logger.debug("Closing open orders", orders=orders)

        for order in orders:
            self.orders.remove(order)

    def get_previous_day_candle(self, contract, days: int = 1, end_date=None):
        csv_path = os.getenv("ALGO_DAILY_ADJUST_CSV")
        if not csv_path:
            raise FileNotFoundError("Set ALGO_DAILY_ADJUST_CSV to a daily-adjust CSV path to use get_previous_day_candle().")

        df = pd.read_csv(str(Path(csv_path).expanduser()))
        df = df[["a_dt", "a_dir", "a_open", "a_high", "a_low", "a_close"]]
        df["a_dt"] = pd.to_datetime(df["a_dt"])

        if end_date is None:
            end_date = df["a_dt"].max()

        end_date -= pd.Timedelta(days=days)
        logger.debug("Getting previous day candle", symbol=contract, end_date=end_date)
        df = df[df["a_dt"].dt.date <= end_date]
        # df = df.sort_values("dt", ascending=False)

        data = df.to_dict(orient="records")
        return data[-1]

    def get_contract_expiration(self, contract: ib_async.Contract):
        pass


class IBKRBroker(BaseBroker):
    """
    Handle every kind of order EXCEPT a STOPLOSS types.
    """

    def __init__(
        self,
        repository: Repository,
        ib_client_id: int = 1,
        port: int = 4001,
        account: str | None = None,
        host: str = "127.0.0.1",
        auto_connect: bool = False,
    ):
        super().__init__(repository)
        self.ib = ib_async.IB()
        self.ib_client_id = ib_client_id
        self.host = host
        self.port = port
        self.account = account
        if auto_connect:
            self.connect()

    def connect(self) -> None:
        if self.ib.isConnected():
            return
        self.ib.connect(self.host, self.port, clientId=self.ib_client_id, account=self.account)

    def disconnect(self) -> None:
        if not self.ib.isConnected():
            return
        self.ib.disconnect()

    def ensure_connected(self) -> None:
        if self.ib.isConnected():
            return
        logger.warning("IB disconnected; reconnecting", host=self.host, port=self.port, client_id=self.ib_client_id)
        self.connect()

    def ib_action(self, action: types.Action):
        if action == types.Action.LONG:
            return "BUY"
        elif action == types.Action.SHORT:
            return "SELL"
        else:
            raise ValueError(f"Unknown action: {action}")

    def get_contract_expiration(self, contract: ib_async.Contract):
        """
        Get the expiration date of a contract.
        """
        self.ensure_connected()
        futures = self.ib.reqContractDetails(contract)[0].contract
        logger.debug("Getting contract", contract=asdict(futures))
        if futures.lastTradeDateOrContractMonth:
            return pd.to_datetime(futures.lastTradeDateOrContractMonth)

        raise ValueError("Contract does not have an expiration date")

    def log_order(self, order: types.BaseOrder):
        """
        call this method AFTER placing an order to prevent IO blocking.
        """
        logger.debug("Order placed", order=asdict(order))
        # self.repository.insert_position(order)

    def _get_contract(self, order: types.BaseOrder):
        self.ensure_connected()
        self.ib.qualifyContracts(order.symbol)
        contract = order.symbol
        return contract

    def _wait_for_order(self, orders):
        """
        Wait for the orders to be submitted.
        """
        if not isinstance(orders, list):
            orders = [orders]

        for order in orders:
            status = getattr(getattr(order, "orderStatus", None), "status", None)
            if status is None:
                logger.debug("Order has no orderStatus; skipping wait", order=order)
                continue
            while "pending" in str(status).lower():
                logger.debug("Waiting for order to be submitted", order=order, status=status)
                self.ib.waitOnUpdate()
                status = getattr(getattr(order, "orderStatus", None), "status", status)

        logger.debug("Order submitted", orders=orders)

    def _limit_order(self, order: types.LimitOrder):
        contract = self._get_contract(order)
        _order = ib_async.LimitOrder(
            action=self.ib_action(order.action),
            totalQuantity=order.quantity,
            lmtPrice=order.price,
            outsideRth=True,
            orderId=self.ib.client.getReqId(),
            tif="GTC",
        )
        open_order = self.ib.placeOrder(contract=contract, order=_order)
        self._wait_for_order(open_order)
        self.log_order(order)
        return order

    def _market_order(self, order: types.MarketOrder):
        contract = self._get_contract(order)
        _order = ib_async.MarketOrder(
            action=self.ib_action(order.action),
            totalQuantity=order.quantity,
            orderId=self.ib.client.getReqId(),
            tif="GTC",
        )
        open_order = self.ib.placeOrder(contract=contract, order=_order)
        self._wait_for_order(open_order)
        self.log_order(order)
        return order

    def _stop_order(self, order: types.StopOrder):
        contract = self._get_contract(order)
        _order = ib_async.StopOrder(
            action=self.ib_action(order.action),
            totalQuantity=order.quantity,
            stopPrice=order.price,
            outsideRth=True,
            orderId=self.ib.client.getReqId(),
            tif="GTC",
        )
        trade = self.ib.placeOrder(contract=contract, order=_order)
        self._wait_for_order(trade)
        self.log_order(order)
        return order

    def _bracket_order(self, order: types.BracketOrder):
        contract = self._get_contract(order)
        qty = order.take_profit.quantity or order.stop_loss.quantity
        if order.take_profit.quantity != order.stop_loss.quantity:
            raise ValueError("Take profit and stop loss orders must have the same quantity")

        orders = self.ib.bracketOrder(
            action=self.ib_action(order.order.action),
            quantity=qty,
            limitPrice=order.order.price,
            takeProfitPrice=order.take_profit.price if order.take_profit else None,
            stopLossPrice=order.stop_loss.price if order.stop_loss else None,
            outsideRth=True,
            tif="GTC",
            account=self.account,
        )

        open_trades = [self.ib.placeOrder(contract=contract, order=_order) for _order in orders]
        self._wait_for_order(open_trades)
        self.log_order(order)
        return open_trades

    def place_oca_exit_orders(
        self,
        contract: ib_async.Contract,
        quantity: int,
        take_profit_price: float,
        stop_loss_price: float,
        action: types.Action,
        oca_group: str,
    ):
        """
        Place OCA-linked protective exit orders (TP+SL).

        Intended for recovery when a position exists but bracket children are missing.
        """
        self.ensure_connected()
        self.ib.qualifyContracts(contract)

        tp_order = ib_async.LimitOrder(
            action=self.ib_action(action),
            totalQuantity=quantity,
            lmtPrice=take_profit_price,
            outsideRth=True,
            orderId=self.ib.client.getReqId(),
            tif="GTC",
        )
        sl_order = ib_async.StopOrder(
            action=self.ib_action(action),
            totalQuantity=quantity,
            stopPrice=stop_loss_price,
            outsideRth=True,
            orderId=self.ib.client.getReqId(),
            tif="GTC",
        )

        tp_order.ocaGroup = oca_group
        sl_order.ocaGroup = oca_group
        tp_order.ocaType = 1
        sl_order.ocaType = 1

        tp_trade = self.ib.placeOrder(contract=contract, order=tp_order)
        sl_trade = self.ib.placeOrder(contract=contract, order=sl_order)
        self._wait_for_order([tp_trade, sl_trade])
        return [tp_trade, sl_trade]

    def to_ib_datetime(self, dt):
        # Warning: You have to convert the timezone to the IB Gateway instance timezone.
        dt = dt.astimezone(pytz.timezone(IB_TZ))
        return dt.strftime("%Y%m%d %H:%M:%S")

    def get_orders(self, contract: ib_async.Contract) -> List[ib_async.Order]:
        self.ensure_connected()
        orders = []
        open_trades = self.ib.openTrades()
        logger.debug("Getting open trades", open_trades=open_trades, open_orders=self.ib.openOrders(), trades=self.ib.trades())
        for trade in open_trades:
            if trade.contract.symbol == contract.symbol:
                orders.append(trade.order)

        return orders

    def get_positions(self, contract: ib_async.Contract = None) -> List[ib_async.Position]:
        self.ensure_connected()
        if contract is None:
            logger.warning("No contract specified, getting all positions")
        positions = []
        for position in self.ib.positions(self.account):
            if contract is None:
                positions.append(position)
            elif position.contract.symbol == contract.symbol:
                positions.append(position)

        return positions

    def get_trades(self, contract: ib_async.Contract) -> List[ib_async.Trade]:
        self.ensure_connected()
        trades = []
        for trade in self.ib.openTrades():
            if trade.contract.symbol == contract.symbol:
                trades.append(trade)

        return trades

    def close_position(self, contract: ib_async.Contract):
        logger.debug("Closing position", symbol=contract)
        positions = self.get_positions(contract)
        for position in positions:
            action = types.Action.SHORT if position.position > 0 else types.Action.LONG
            order = types.MarketOrder(
                symbol=contract.symbol,
                quantity=abs(position.position),
                action=action,
                timestamp=utils.get_now(),
            )
            self._market_order(order)

    def cancel_orders(self, orders: List[ib_async.Order]):
        self.ensure_connected()
        logger.debug("Closing open orders", orders=orders)
        for order in orders:
            self.ib.cancelOrder(order)

    def get_position_prices(self, contract: ib_async.Contract = None):
        """
        Get the entry_price, take profit and stop loss of a position.
        """
        self.ensure_connected()
        positions = self.get_positions(contract)
        if not positions:
            raise ValueError(f"No positions found for contract: {contract.symbol}")

        if len(positions) > 1:
            raise ValueError(f"Multiple positions found for contract: {contract.symbol}.")

        entry_price = positions[0].avgCost

        # trades = self.get_trades(contract)
        # if not trades:
        #     raise ValueError(f"No trades found for contract: {contract.symbol}")

        # orders = [trade.order for trade in trades]
        orders = self.get_orders(contract)
        if not orders:
            raise ValueError(f"No orders found for contract: {contract.symbol}")

        target_price = None
        stop_loss_price = None
        action = None
        for order in orders:
            if order.orderType == "LMT":
                target_price = order.lmtPrice

            elif order.orderType == "STP":
                stop_loss_price = order.auxPrice

            if order.action == "BUY":
                action = types.Action.SHORT
            elif order.action == "SELL":
                action = types.Action.LONG

        return {
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss_price": stop_loss_price,
            "action": action,
        }

    def modify_position_stoploss(self, contract: ib_async.Contract, stop_loss: float):
        """
        Modify the stop loss of a position.
        """
        self.ensure_connected()
        positions = self.get_positions(contract)
        if not positions:
            raise ValueError(f"No positions found for contract: {contract.symbol}")

        if len(positions) > 1:
            raise ValueError(f"Multiple positions found for contract: {contract.symbol}.")

        trades = self.get_trades(contract)
        if not trades:
            raise ValueError(f"No trades found for contract: {contract.symbol}")

        orders = [trade.order for trade in trades]
        if not orders:
            raise ValueError(f"No orders found for contract: {contract.symbol}")

        for order in orders:
            if order.orderType == "STP":
                old_stop_loss = order.auxPrice
                order.auxPrice = stop_loss
                t = self.ib.placeOrder(contract=contract, order=order)
                self._wait_for_order(t)
                logger.debug("Stop loss modified", old_stop_loss=old_stop_loss, new_stop_loss=stop_loss, order=order)
                return

        raise ValueError(f"No stop loss order found for contract: {contract.symbol}")

    def close_all_positions(self, contract: ib_async.Contract = None):
        logger.info("Closing all positions", contract=contract)

        positions = self.get_positions(contract=contract)
        if not positions:
            logger.debug("No positions to close", contract=contract)
            return

        for position in positions:
            contract = position.contract
            orders = self.get_orders(contract=contract)
            for order in orders:
                trade = self.ib.cancelOrder(order)
                self._wait_for_order(trade)

            for position in positions:
                if position.position != 0:
                    action = types.Action.SHORT if position.position > 0 else types.Action.LONG
                    order = types.MarketOrder(
                        symbol=position.contract,
                        quantity=abs(position.position),
                        action=action,
                        timestamp=None,
                    )
                    self._market_order(order)
