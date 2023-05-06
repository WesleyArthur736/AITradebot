import ta
import ccxt
import math
from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np
import math


class RSI_Trading_Bot:
    def __init__(self, overbought, oversold, window):
        self.OVERBOUGHT = overbought
        self.OVERSOLD = oversold
        self.WINDOW = window

    def check_rsi_position(self, trade_signals):
        trade_signals['current_rsi_position'] = float("nan")
        trade_signals.loc[trade_signals['rsi'] >
                          self.OVERBOUGHT, 'current_rsi_position'] = 2
        trade_signals.loc[(trade_signals['rsi'] > 0) & (trade_signals['rsi'] <
                          self.OVERSOLD), 'current_rsi_position'] = 1
        trade_signals.loc[(trade_signals['rsi'] >= self.OVERSOLD) & (trade_signals['rsi'] <=
                          self.OVERBOUGHT), 'current_rsi_position'] = 0

    def get_rsi_signal(self, ohlcv_df):
        trade_signals = ohlcv_df.copy()
        rsi_indicator = RSIIndicator(
            trade_signals['close'], window=self.WINDOW)
        trade_signals['rsi'] = rsi_indicator.rsi()

        trade_signals['trade_signal'] = 0
        trade_signals['rsi'].fillna(0, inplace=True)
        self.check_rsi_position(trade_signals)
        trade_signals.loc[(trade_signals['current_rsi_position'] != -1) & (trade_signals['current_rsi_position'].shift(-1) != -1),
                          'change'] = trade_signals['current_rsi_position'].shift(1) - trade_signals['current_rsi_position']
        trade_signals.loc[trade_signals['change'] ==
                          2, 'trade_signal'] = -1  # sell signal
        trade_signals.loc[trade_signals['change']
                          == 1, 'trade_signal'] = 1  # buy signal

        # Drop the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(
            columns=['current_rsi_position', 'change', 'rsi'])
        return trade_signals
