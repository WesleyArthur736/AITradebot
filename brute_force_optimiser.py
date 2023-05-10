import trader_bots 
import utils

ohlcv_df = utils.get_daily_ohlcv_data()

fee_percentage = 0.02

MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14, 'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}
SAR_parameters = {'bot_name': 'SAR_bot', 'step': 0.02, 'max_step': 0.2}
OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
ROC_parameters = {'bot_name': 'ROC_bot', 'window': 12, 'buy_threshold': 5, 'sell_threshold': -5}

all_parameters = [
    MACD_parameters, 
    Bollinger_Bands_parameters, 
    RSI_parameters, 
    VWAP_parameters, 
    Stochastic_Oscillator_parameters,
    OBV_trend_following_parameters,
    SAR_parameters,
    OBV_trend_reversal_parameters,
    ROC_parameters
    ]
target = 120
attempts_limit = 100
final_balance = 0
attempts = 0

while final_balance <= target and attempts < attempts_limit:
    attempts += 1
    print(f"Attempt number {attempts}")
    buy_dnf, sell_dnf, trade_signals = trader_bots.ensemble_bot(
            ohlcv_df = ohlcv_df,
            all_parameters = all_parameters,
            min_literals = 1,
            max_literals = 3,
            min_conjunctions = 1,
            max_conjunctions = 5 
        ).generate_signals()

    final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Final balance: {final_balance}")

if final_balance > target:
    print(f"After {attempts} attempts, we have a winner!")
    print()
    print("Buy DNF:")
    print(buy_dnf)
    print()
    print("Sell DNF:")
    print(sell_dnf)
    print()
    print("Final Balance:")
    print(final_balance)
    print()

    utils.plot_trading_simulation(trade_results, "Ensemble")   

else:
    print(f"Failed to produce a final balance of {target}.")