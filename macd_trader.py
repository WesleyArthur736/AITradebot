import numpy as np
from ta.trend import MACD
from models import Trader


class MACDTrader(Trader):
    def generate_signals(self, ohlcv, **kwargs):
        macd_indicator = MACD(ohlcv["Mid Price"])

        # TODO rename this first signals var
        signals = np.where(macd_indicator.macd() > macd_indicator.macd_signal(), 1.0, 0.0)
        signals = np.diff(signals)
        signals = np.pad(signals, (1,), 'constant', constant_values=0)
        return signals
