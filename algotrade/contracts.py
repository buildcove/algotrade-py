from dataclasses import dataclass
from typing import Dict

import ib_async

from .core import types


@dataclass(frozen=True)
class ContractSpecification:
    symbol: str
    exchange: str
    tick_size: float
    tick_value: float
    tick_spread: int
    ib_symbol: str
    ib_exchange: str
    ib_local_symbol: str

    @property
    def ib(self) -> ib_async.Future:
        return ib_async.Future(symbol=self.ib_symbol, exchange=self.ib_exchange, localSymbol=self.ib_local_symbol)


CONTRACT_SPECIFICATIONS: Dict[types.SupportedContract, ContractSpecification] = {
    types.SupportedContract.MES: ContractSpecification(
        symbol="MES",
        exchange="CME_MINI",
        tick_size=0.25,
        tick_value=1.25,
        tick_spread=0,
        ib_symbol="MES",
        ib_exchange="CME",
        ib_local_symbol="MESZ5",
    ),
    types.SupportedContract.MGC: ContractSpecification(
        symbol="MGC",
        exchange="COMEX_MINI",
        tick_size=0.10,
        tick_value=1.00,
        tick_spread=0,
        ib_symbol="MGC",
        ib_exchange="COMEX",
        ib_local_symbol="MGCZ5",
    ),
    types.SupportedContract.PL: ContractSpecification(
        symbol="PL",
        exchange="NYMEX",
        tick_size=0.10,
        tick_value=5.00,
        tick_spread=0,
        ib_symbol="PL",
        ib_exchange="NYMEX",
        ib_local_symbol="PLV5",
    ),
    types.SupportedContract.MHG: ContractSpecification(
        symbol="MHG",
        exchange="COMEX_MINI",
        tick_size=0.0005,
        tick_value=1.25,
        tick_spread=2,
        ib_symbol="MHG",
        ib_exchange="COMEX",
        ib_local_symbol="MHGU5",
    ),
    types.SupportedContract.MCL: ContractSpecification(
        symbol="MCL",
        exchange="NYMEX",
        tick_size=0.01,
        tick_value=1.00,
        tick_spread=0,
        ib_symbol="MCL",
        ib_exchange="NYMEX",
        ib_local_symbol="MCLV5",
    ),
    types.SupportedContract.CL: ContractSpecification(
        symbol="CL",
        exchange="NYMEX",
        tick_size=0.01,
        tick_value=10.00,
        tick_spread=0,
        ib_symbol="CL",
        ib_exchange="NYMEX",
        ib_local_symbol="CLV5",
    ),
    types.SupportedContract.MNG: ContractSpecification(
        symbol="MNG",
        exchange="NYMEX",
        tick_size=0.001,
        tick_value=1.00,
        tick_spread=0,
        ib_symbol="MHNG",
        ib_exchange="NYMEX",
        ib_local_symbol="MNGU25",
    ),
    types.SupportedContract.NG: ContractSpecification(
        symbol="NG",
        exchange="NYMEX",
        tick_size=0.001,
        tick_value=10.00,
        tick_spread=0,
        ib_symbol="NG",
        ib_exchange="NYMEX",
        ib_local_symbol="NGU25",
    ),
    types.SupportedContract.M6E: ContractSpecification(
        symbol="M6E",
        exchange="CME_MINI",
        tick_size=0.0001,
        tick_value=1.25,
        tick_spread=2,
        ib_symbol="M6E",
        ib_exchange="CME",
        ib_local_symbol="M6EM5",
    ),
    types.SupportedContract.JPY6J: ContractSpecification(
        symbol="6J",
        exchange="CME",
        tick_size=0.000001,
        tick_value=12.50,
        tick_spread=0,
        ib_symbol="6J",
        ib_exchange="CME",
        ib_local_symbol="6JU5",
    ),
    types.SupportedContract.GBP6B: ContractSpecification(
        symbol="6B",
        exchange="CME",
        tick_size=0.0001,
        tick_value=6.25,
        tick_spread=0,
        ib_symbol="6B",
        ib_exchange="CME",
        ib_local_symbol="6BU5",
    ),
    types.SupportedContract.ZM: ContractSpecification(
        symbol="ZM",
        exchange="CBOT",
        tick_size=0.1,
        tick_value=10.00,
        tick_spread=0,
        ib_symbol="ZM",
        ib_exchange="CBOT",
        ib_local_symbol="ZMZ5",
    ),
}


def get_contract_spec(contract: types.SupportedContract) -> ContractSpecification:
    return CONTRACT_SPECIFICATIONS[contract]


ContractSpec = ContractSpecification
