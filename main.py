import bot
import trade
import pandas as pd


def main():
    pd.set_option('display.max_rows', None)

    ohlcv_df = trade.get_data()

    print("RSI Trading Bot")
    trade_signals = bot.RSI_Trading_Bot(70, 30, 14).get_rsi_signal(ohlcv_df)
    final_balance, trade_results = trade.execute_trades(trade_signals, 0.02)
    print("RSI Trade Signals:")
    print(trade_results)
    print("RSI Final Balance:")
    print(final_balance)
    print()


main()
