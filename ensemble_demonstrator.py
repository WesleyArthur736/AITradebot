import functions 

def main():
    # Get data
    ohlcv_df = functions.Simulate.get_daily_ohlcv_data()
    
    # Define parameter values (ultimately, determine their optimal values).
    MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
    Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
    RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
    VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
    Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14, 'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}

    all_parameters = [MACD_parameters, Bollinger_Bands_parameters, RSI_parameters, VWAP_parameters, Stochastic_Oscillator_parameters]


    # ^ These parameters should be determined by the GA. Once they've been optimised, they can be fed to the ensembler.

    all_bot_signals = functions.Ensemble(ohlcv_df).determine_bot_signals(all_parameters)

    print(all_bot_signals)

if __name__ == "__main__":
    main()

