import trader_bots 
import utils

def main(fee_percentage):
    ohlcv_df = utils.get_daily_ohlcv_data()

    # print("MACD Trading Bot")
    # trade_signals = trader_bots.MACD_bot(
    #     ohlcv_df = ohlcv_df,
    #     slow_window = 26, 
    #     fast_window = 12, 
    #     signal_window = 9
    # ).generate_signals()
    # print()
    # print("MACD Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("MACD Trade Results:")
    # print(trade_results)
    # print("MACD Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "MACD")

    # print("Bollinger Bands Trading Bot")
    # trade_signals = trader_bots.bollinger_bands_bot(
    #     ohlcv_df = ohlcv_df, 
    #     window = 20, 
    #     num_standard_deviations = 2.5
    # ).generate_signals()
    # print()
    # print("Bollinger Bands Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("Bollinger Bands Trade Results:")
    # print(trade_results)
    # print("Bollinger Bands Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "Bollinger Band")

    # print("RSI Trading Bot")
    # trade_signals = trader_bots.RSI_bot(
    #     ohlcv_df = ohlcv_df, 
    #     overbought_threshold = 70, 
    #     oversold_threshold = 30, 
    #     window = 14
    # ).generate_signals()
    # print()
    # print("RSI Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("RSI Trade Results:")
    # print(trade_results)
    # print("RSI Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "RSI")

    # print("VWAP Trading Bot")
    # trade_signals = trader_bots.VWAP_bot(
    #     ohlcv_df = ohlcv_df, 
    #     window = 20
    # ).generate_signals()
    # print()
    # print("VWAP Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("VWAP Trade Results:")
    # print(trade_results)
    # print("VWAP Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "VWAP")

    # print("Stochastic Oscillator Trading Bot")
    # trade_signals = trader_bots.stochastic_oscillator_bot(
    #     ohlcv_df = ohlcv_df, 
    #     oscillator_window = 14, 
    #     signal_window = 3, 
    #     overbought_threshold = 80, 
    #     oversold_threshold = 20
    # ).generate_signals()
    # print()
    # print("Stochastic Oscillator Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("Stochastic Oscillator Trade Results:")
    # print(trade_results)
    # print("Stochastic Oscillator Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "Stochastic Oscillator")

    print("Ensemble Trading Bot")
    buy_dnf, sell_dnf, trade_signals = trader_bots.ensemble_bot(
        ohlcv_df = ohlcv_df,
        all_parameters = all_parameters,
        min_literals = 1,
        max_literals = 5,
        min_conjunctions = 1,
        max_conjunctions = 4 
    ).generate_signals()
    print()
    print("Buy DNF:")
    print(buy_dnf)
    print()
    print("Sell DNF:")
    print(sell_dnf)
    print()
    print("Ensemble Trade Signals:")
    print(trade_signals)
    print()
    final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    print("Ensemble Trade Results:")
    print(trade_results)
    print("Ensemble Final Balance:")
    print(final_balance)
    print()
    utils.plot_trading_simulation(trade_results, "Ensemble")    

if __name__ == "__main__":

    fee_percentage = 0.02

    MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
    Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
    RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
    VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
    Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14, 'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}

    all_parameters = [MACD_parameters, Bollinger_Bands_parameters, RSI_parameters, VWAP_parameters, Stochastic_Oscillator_parameters]

    main(fee_percentage)
