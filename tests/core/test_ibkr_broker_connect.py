import pytest

pytest.importorskip("ib_async")

import ib_async
from algotrade.core import brokers, repository


def test_ibkr_broker_does_not_connect_on_init(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def fake_connect(self, host, port, clientId, account=None):  # noqa: ANN001, N803 - match IB signature
        called.append((host, port, clientId, account))

    monkeypatch.setattr(ib_async.IB, "connect", fake_connect)

    repo = repository.Repository()
    broker = brokers.IBKRBroker(repo, port=4002, account="DU123", ib_client_id=7)
    assert called == []

    broker.connect()
    assert called == [("127.0.0.1", 4002, 7, "DU123")]
