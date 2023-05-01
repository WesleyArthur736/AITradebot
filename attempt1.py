import ccxt
import pandas as pd
import ta
import numpy as np
from datetime import datetime, timedelta
from ta.trend import MACD



def get_data():
    exchange = ccxt.kraken()
    ohlcv_data = exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
    ohlcv_df = pd.DataFrame(ohlcv_data, columns = ["timestamp","open", "high", "low", "close", "volume"])

    return ohlcv_df


def export_data():
    print()



def import_data():
    print()



def determine_triggers(ohlcv_df):
    close_prices = ohlcv_df['close']

    # Calculate MACD, signal line, and histogram
    macd_indicator = MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
    ohlcv_df['macd'] = macd_indicator.macd()
    ohlcv_df['signal'] = macd_indicator.macd_signal()
    ohlcv_df['histogram'] = macd_indicator.macd_diff()

    # Determine buy and sell signals
    ohlcv_df['signal_flag'] = 0
    ohlcv_df.loc[ohlcv_df['histogram'] > 0, 'signal_flag'] = 1
    ohlcv_df.loc[ohlcv_df['histogram'] < 0, 'signal_flag'] = -1
    ohlcv_df['trading_signal'] = ohlcv_df['signal_flag'].diff()

    # Find buy and sell signals in the DataFrame
    buy_signals = ohlcv_df[ohlcv_df['trading_signal'] == 2]  # 2 = transition from -1 to 1
    sell_signals = ohlcv_df[ohlcv_df['trading_signal'] == -2]  # -2 = transition from 1 to -1

    # Makes sure first trade trigger is a buy
    if not sell_signals.empty and not buy_signals.empty:
        if sell_signals.index[0] < buy_signals.index[0]:
            sell_signals = sell_signals.iloc[1:]

    

    # Merged trade signals (alternates buy/sell)
    trade_signals = pd.concat([buy_signals, sell_signals])
    trade_signals.sort_index(inplace=True)

    return trade_signals



def execute_trades(trade_signals, AUD_BALANCE, BTC_BALANCE, fee_percentage, ohlcv_df):
    # trades_completed = 0
    # num_trades = len(buy_signals) + len(sell_signals)

    # while trades_completed < num_trades:
    #     if (trades_completed % 2) == 0:

    for  index, row in trade_signals.iterrows():
        trading_signal = row['trading_signal']
    
        if trading_signal == 2:  # Buy signal
            # Convert all AUD to BTC using the close price
            BTC_BALANCE = AUD_BALANCE / row['close'] * (1 - fee_percentage)
            AUD_BALANCE = 0

        elif trading_signal == -2:  # Sell signal
            # Convert all BTC to AUD using the close price
            AUD_BALANCE = BTC_BALANCE * row['close'] * (1 - fee_percentage)
            BTC_BALANCE = 0

    if AUD_BALANCE == 0:
        last_close_price = ohlcv_df['close'].iloc[-1]
        AUD_BALANCE = BTC_BALANCE * last_close_price * (1 - fee_percentage)
        BTC_BALANCE = 0


    # Print the final balance
    return AUD_BALANCE, BTC_BALANCE



def evaluate_performance():
    print()



def main():

    AUD_BALANCE = 100.00
    BTC_BALANCE = 0.00

    ohlcv_df = get_data()
    trade_signals = determine_triggers(ohlcv_df)
    AUD_BALANCE, BTC_BALANCE = execute_trades(trade_signals, AUD_BALANCE, BTC_BALANCE, 0.02, ohlcv_df)

    print(AUD_BALANCE, BTC_BALANCE)



main()