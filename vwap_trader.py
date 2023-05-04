import numpy as np
from ta.volume import VolumeWeightedAveragePrice
from models import Trader


class VWAPTrader(Trader):
    def __init__(self, **kwargs):
        self.vwap_window = kwargs["vwap_window"]
    def generate_signals(self, ohlcv):
        vwap_indicator = VolumeWeightedAveragePrice(high=ohlcv["High"],
                                                    low=ohlcv["Low"],
                                                    close=ohlcv["Close"],
                                                    volume=ohlcv["Volume"],
                                                    window=self.vwap_window)

        # # TODO rename this first signals var
        # signals = np.where(macd_indicator.macd() > macd_indicator.macd_signal(), 1.0, 0.0)
        # signals = np.diff(signals)
        # signals = np.pad(signals, (1,), 'constant', constant_values=0)

        x = vwap_indicator.volume_weighted_average_price()

        return np.pad(np.diff(np.where(x > ohlcv["Close"], 1.0, 0.0)), (1,), 'constant', constant_values=0)
