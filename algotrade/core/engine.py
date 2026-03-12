from algotrade.core import brokers


class BaseEngine:
    def __init__(self, broker: brokers.BaseBroker):
        self.broker = broker
