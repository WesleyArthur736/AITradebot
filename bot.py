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

    # find the current rsi position
    # add 'current_rsi_position into the df
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

        # initialise output columns.
        trade_signals['buy_signal'] = False
        trade_signals['sell_signal'] = False

        trade_signals['trade_signal'] = 0
        trade_signals['rsi'].fillna(0, inplace=True)

        self.check_rsi_position(trade_signals)

        # start from 14 as the first 14 days do not have an rsi value
        for i in range(14, trade_signals.shape[0]):
            change = trade_signals['current_rsi_position'][i -
                                                           1] - trade_signals['current_rsi_position'][i]
            # evaluate literals
            sell_signal = change > 1  # if change == 2, sell
            buy_signal = change > 0  # if change == 1, buy

            # find the dnf and get the signals
            trade_signals.at[i, 'buy_signal'] = buy_signal and not sell_signal
            trade_signals.at[i, 'sell_signal'] = sell_signal and buy_signal

        # Drop the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(
            columns=['current_rsi_position', 'rsi'])
        return trade_signals
