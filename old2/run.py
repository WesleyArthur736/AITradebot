import functions 

def main():
    ohlcv_df = functions.get_daily_ohlcv_data()

    # print("MACD Trading Bot")
    # trade_signals = functions.Bots(ohlcv_df).MACD_bot(slow_window = 26, fast_window = 12, signal_window = 9)
    # print()
    # print("MACD Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = functions.execute_trades(trade_signals, 0.02)
    # print("MACD Trade Signals:")
    # print(trade_results)
    # print("MACD Final Balance:")
    # print(final_balance)
    # print()
    # functions.plot_trading_simulation(trade_results)

    # print("Bollinger Bands Trading Bot")
    # trade_signals = functions.Bots(ohlcv_df).bollinger_bands_bot(window = 20, num_standard_deviations = 2.5)
    # print()
    # print("Bollinger Bands Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = functions.execute_trades(trade_signals, 0.02)
    # print("Bollinger Bands Trade Signals:")
    # print(trade_results)
    # print("Bollinger Bands Final Balance:")
    # print(final_balance)
    # print()
    # functions.plot_trading_simulation(trade_results)

    # print("RSI Trading Bot")
    # trade_signals = functions.Bots(ohlcv_df).RSI_bot(overbought_threshold = 70, oversold_threshold = 30, window = 14)
    # print()
    # print("RSI Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = functions.execute_trades(trade_signals, 0.02)
    # print("RSI Trade Signals:")
    # print(trade_results)
    # print("RSI Final Balance:")
    # print(final_balance)
    # print()
    # functions.plot_trading_simulation(trade_results)

    # print("VWAP Trading Bot")
    # trade_signals = functions.Bots(ohlcv_df).VWAP_bot(window = 20)
    # print()
    # print("VWAP Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = functions.execute_trades(trade_signals, 0.02)
    # print("VWAP Trade Signals:")
    # print(trade_results)
    # print("VWAP Final Balance:")
    # print(final_balance)
    # print()
    # functions.plot_trading_simulation(trade_results)

    print("Stochastic Oscillator Trading Bot")
    trade_signals = functions.Bots(ohlcv_df).stochastic_oscillator_bot(oscillator_window = 14, signal_window = 3, overbought_threshold = 80, oversold_threshold = 20)
    print()
    print("Stochastic Oscillator Trade Signals:")
    print(trade_signals)
    print()
    final_balance, trade_results = functions.execute_trades(trade_signals, 0.02)
    print("Stochastic Oscillator Trade Signals:")
    print(trade_results.to_string())
    print("Stochastic Oscillator Final Balance:")
    print(final_balance)
    print()
    functions.plot_trading_simulation(trade_results)

main()
