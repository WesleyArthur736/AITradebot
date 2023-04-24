import numpy as np
from ta.trend import MACD
from models import Trader


class MACDTrader(Trader):
    def __init__(self, **kwargs):
        self.macd_window_slow = kwargs["window_slow"]
        self.macd_window_fast = kwargs["window_fast"]
    def generate_signals(self, ohlcv):
        macd_indicator = MACD(ohlcv["Mid Price"], window_slow=self.macd_window_slow, window_fast=self.macd_window_fast)

        # TODO rename this first signals var
        signals = np.where(macd_indicator.macd() > macd_indicator.macd_signal(), 1.0, 0.0)
        signals = np.diff(signals)
        signals = np.pad(signals, (1,), 'constant', constant_values=0)
        return signals
