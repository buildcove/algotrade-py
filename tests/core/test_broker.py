from algotrade.core import brokers, repository, types


def test_broker_bracket_order_take_profit():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo)
    order = types.BracketOrder(
        symbol="AAPL",
        order=types.MarketOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.LONG,
            timestamp="2021-01-01 00:00:00",
        ),
        take_profit=types.LimitOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.SHORT,
            price=150.0,
            timestamp="2021-01-01 00:00:00",
        ),
        stop_loss=types.StopOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.SHORT,
            price=100.0,
            timestamp="2021-01-01 00:00:00",
        ),
    )
    _broker.order(order)
    assert len(_broker.orders) == 3
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=120.0, ask=120.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 2
    assert len(_broker.positions) == 1
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=150.0, ask=150.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 1

    assert _broker.closed_positions[0].quantity == 1
    assert _broker.closed_positions[0].action == types.Action.LONG
    assert _broker.closed_positions[0].entry_price == 120.0
    assert _broker.closed_positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.closed_positions[0].exit_price == 150.0
    assert _broker.closed_positions[0].symbol == "AAPL"


def test_broker_bracket_order_stop_loss():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo)
    order = types.BracketOrder(
        symbol="AAPL",
        order=types.MarketOrder(symbol="AAPL", quantity=1, action=types.Action.LONG, timestamp="2021-01-01 00:00:00"),
        take_profit=types.LimitOrder(
            symbol="AAPL", quantity=1, action=types.Action.SHORT, price=150.0, timestamp="2021-01-01 00:00:00"
        ),
        stop_loss=types.StopOrder(
            symbol="AAPL", quantity=1, action=types.Action.SHORT, price=100.0, timestamp="2021-01-01 00:00:00"
        ),
    )
    _broker.order(order)

    assert len(_broker.orders) == 3
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=90.0, ask=90.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 1

    assert _broker.closed_positions[0].quantity == 1
    assert _broker.closed_positions[0].action == types.Action.LONG
    assert _broker.closed_positions[0].entry_price == 90.0
    assert _broker.closed_positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.closed_positions[0].exit_price == 90.0
    assert _broker.closed_positions[0].symbol == "AAPL"


def test_broker_bracket_limit_order():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo)
    order = types.BracketOrder(
        symbol="AAPL",
        order=types.LimitOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.LONG,
            price=100.0,
            timestamp="2021-01-01 00:00:00",
        ),
        take_profit=types.LimitOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.SHORT,
            price=150.0,
            timestamp="2021-01-01 00:00:00",
        ),
        stop_loss=types.StopOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.SHORT,
            price=50.0,
            timestamp="2021-01-01 00:00:00",
        ),
    )
    _broker.order(order)

    assert len(_broker.orders) == 3
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=90.0, ask=90.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)

    assert len(_broker.orders) == 3
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=100.0, ask=100.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 2
    assert len(_broker.positions) == 1
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=150.0, ask=150.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 1

    assert _broker.closed_positions[0].quantity == 1
    assert _broker.closed_positions[0].action == types.Action.LONG
    assert _broker.closed_positions[0].entry_price == 100.0
    assert _broker.closed_positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.closed_positions[0].exit_price == 150.0
    assert _broker.closed_positions[0].symbol == "AAPL"


def test_broker_limit_order_with_spread():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo, tick_size=0.1, limit_spread_ticks=1)
    order = types.LimitOrder(
        symbol="AAPL",
        quantity=1,
        action=types.Action.LONG,
        price=100.0,
        timestamp="2021-01-01 00:00:00",
    )
    _broker.order(order)

    assert len(_broker.orders) == 1
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=99.0, ask=99.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 1
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=100.1, ask=100.1, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 1
    assert len(_broker.closed_positions) == 0

    assert _broker.positions[0].quantity == 1
    assert _broker.positions[0].action == types.Action.LONG
    assert _broker.positions[0].entry_price == 100.0
    assert _broker.positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.positions[0].symbol == "AAPL"


def test_broker_stop_order_long():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo)
    order = types.StopOrder(
        symbol="AAPL",
        quantity=1,
        action=types.Action.LONG,
        price=100.0,
        timestamp="2021-01-01 00:00:00",
    )
    _broker.order(order)

    assert len(_broker.orders) == 1
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=99.0, ask=99.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 1
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=100.1, ask=100.1, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 1
    assert len(_broker.closed_positions) == 0

    assert _broker.positions[0].quantity == 1
    assert _broker.positions[0].action == types.Action.LONG
    assert _broker.positions[0].entry_price == 100.1
    assert _broker.positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.positions[0].symbol == "AAPL"


def test_broker_stop_order_short():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo)
    order = types.StopOrder(
        symbol="AAPL",
        quantity=1,
        action=types.Action.SHORT,
        price=100.0,
        timestamp="2021-01-01 00:00:00",
    )
    _broker.order(order)

    assert len(_broker.orders) == 1
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=101.0, ask=101.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 1
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=99.9, ask=99.9, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 1
    assert len(_broker.closed_positions) == 0

    assert _broker.positions[0].quantity == 1
    assert _broker.positions[0].action == types.Action.SHORT
    assert _broker.positions[0].entry_price == 99.9
    assert _broker.positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.positions[0].symbol == "AAPL"


def test_broker_brack_order_gain():
    repo = repository.Repository()
    _broker = brokers.BacktestingBroker(repo, tick_size=1, limit_spread_ticks=1)
    order = types.BracketOrder(
        symbol="AAPL",
        order=types.MarketOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.LONG,
            timestamp="2021-01-01 00:00:00",
        ),
        take_profit=types.LimitOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.SHORT,
            price=150.0,
            timestamp="2021-01-01 00:00:00",
        ),
        stop_loss=types.StopOrder(
            symbol="AAPL",
            quantity=1,
            action=types.Action.SHORT,
            price=100.0,
            timestamp="2021-01-01 00:00:00",
        ),
    )
    _broker.order(order)

    assert len(_broker.orders) == 3
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=120.0, ask=120.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 2
    assert len(_broker.positions) == 1
    assert len(_broker.closed_positions) == 0

    stream = types.Stream(bid=150.0, ask=150.0, timestamp="2021-01-01 00:00:00")
    _broker.next(stream)
    assert len(_broker.orders) == 0
    assert len(_broker.positions) == 0
    assert len(_broker.closed_positions) == 1

    assert _broker.closed_positions[0].quantity == 1
    assert _broker.closed_positions[0].action == types.Action.LONG
    assert _broker.closed_positions[0].entry_price == 120.0
    assert _broker.closed_positions[0].entry_timestamp == "2021-01-01 00:00:00"
    assert _broker.closed_positions[0].exit_price == 150.0
    assert _broker.closed_positions[0].symbol == "AAPL"
    assert _broker.closed_positions[0].gain_ticks == 30.0
