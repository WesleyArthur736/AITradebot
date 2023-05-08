import bots 

def main():
    ohlcv_df = bots.get_daily_ohlcv_data()

    print(ohlcv_df)

    print("MACD Trading Bot")
    trade_signals = bots.MACD_Trading_Bot(slow_window = 26, fast_window = 12, signal_window = 9).determine_MACD_signals(ohlcv_df)
    print()
    print("MACD Trade Signals:")
    print(trade_signals.to_string())
    print()
    final_balance, trade_results = bots.execute_trades(trade_signals, 0.00)
    print("MACD Trade Signals:")
    print(trade_results)
    print("MACD Final Balance:")
    print(final_balance)
    print()
    bots.plot_trading_simulation(trade_results)

    print("Bollinger Bands Trading Bot")
    trade_signals = bots.Bollinger_Bands_Trading_Bot(window = 20, num_standard_deviations = 2.5).determine_BB_signals(ohlcv_df)
    print()
    print("Bollinger Bands Trade Signals:")
    print(trade_signals)
    print()
    final_balance, trade_results = bots.execute_trades(trade_signals, 0.00)
    print("Bollinger Bands Trade Signals:")
    print(trade_results)
    print("Bollinger Bands Final Balance:")
    print(final_balance)
    print()
    bots.plot_trading_simulation(trade_results)

main()
