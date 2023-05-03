import ta
import ccxt
import math
from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np


def get_data():
    exchange = ccxt.kraken()
    bars = exchange.fetch_ohlcv('BTC/AUD', timeframe="1d", limit=720)
    df = pd.DataFrame(
        bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df


def check_position(rsi):
    # sell when rsi decrease, below 70
    RSI_OVERBOUGHT = 70
    # buy when rsi increase, passing 30
    RSI_OVERSOLD = 30

    if rsi > RSI_OVERBOUGHT:
        return 2
    elif rsi < RSI_OVERSOLD:
        return 1
    else:
        return 0


def trade_rsi(aud, btc, ohlcv):
    starting_index = 0
    initial_position = 0
    FEES_PERCENTAGE = 1-0.02

    rsi_indicator = RSIIndicator(ohlcv['close'], window=14)
    ohlcv['rsi'] = rsi_indicator.rsi()

    # finding the initial position
    for i in range(ohlcv.shape[0]):
        current_rsi = ohlcv['rsi'][i]
        if not math.isnan(current_rsi):
            initial_position = check_position(current_rsi)
            starting_index = i+1
            break

    for i in range(starting_index, ohlcv.shape[0]):
        current_rsi = ohlcv['rsi'][i]
        current_close = ohlcv['close'][i]
        current_position = check_position(current_rsi)
        change = current_position - initial_position
        if change == 2 and btc > 0:
            aud = btc * current_close * (FEES_PERCENTAGE)
            btc = 0
            print("Sell", aud, btc)
        elif change == 1 and aud > 0:
            btc = aud / current_close * (FEES_PERCENTAGE)
            aud = 0
            print("Buy", aud, btc)

        initial_position = current_position

    return aud, btc


def main():
    AUD_BALANCE = 100.00
    BTC_BALANCE = 0

    ohlcv = get_data()
    AUD_BALANCE, BTC_BALANCE = trade_rsi(AUD_BALANCE, BTC_BALANCE, ohlcv)
    print(AUD_BALANCE, BTC_BALANCE)


main()
