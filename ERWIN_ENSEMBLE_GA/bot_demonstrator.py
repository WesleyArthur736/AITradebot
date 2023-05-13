import trader_bots 
import utils
from sklearn.model_selection import train_test_split

def main(fee_percentage):
    ohlcv_df = utils.get_daily_ohlcv_data()
    ohlcv_df_train, ohlcv_df_test = train_test_split(ohlcv_df, test_size = 0.2, shuffle = False)

### MACD ###

    print("MACD Trading Bot")
    trade_signals = trader_bots.MACD_bot(
        ohlcv_df = ohlcv_df_test,
        slow_window = 26, 
        fast_window = 12, 
        signal_window = 9
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Generic MACD Final Balance: {final_balance}")

    trade_signals = trader_bots.MACD_bot(
        ohlcv_df = ohlcv_df_test,
        slow_window = 8, 
        fast_window = 3, 
        signal_window = 2
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised MACD Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic MACD", "Optimised MACD"], "Effect of GA Parameter Optimisation on MACD Bot Performance")


### Bollinger Bands ###

    print("Bollinger Bands Trading Bot")
    trade_signals = trader_bots.bollinger_bands_bot(
        ohlcv_df = ohlcv_df_test, 
        window = 20, 
        num_standard_deviations = 2
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Bollinger Bands Final Balance: {final_balance}")

    trade_signals = trader_bots.bollinger_bands_bot(
        ohlcv_df = ohlcv_df_test, 
        window = 15, 
        num_standard_deviations = 2.5
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised Bollinger Bands Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic Bollinger Bands", "Optimised Bollinger Bands"], "Effect of GA Parameter Optimisation on Bollinger Bands Bot Performance")


### RSI ###

    print("RSI Trading Bot")
    trade_signals = trader_bots.RSI_bot(
        ohlcv_df = ohlcv_df_test, 
        overbought_threshold = 70, 
        oversold_threshold = 30, 
        window = 14
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Generic RSI Final Balance: {final_balance}")

    trade_signals = trader_bots.RSI_bot(
        ohlcv_df = ohlcv_df_test, 
        overbought_threshold = 60, 
        oversold_threshold = 40, 
        window = 12
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(F"Optimised RSI Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic RSI", "Optimised RSI"], "Effect of GA Parameter Optimisation on RSI Bot Performance")


### VWAP ###

    print("VWAP Trading Bot")
    trade_signals = trader_bots.VWAP_bot(
        ohlcv_df = ohlcv_df_test, 
        window = 20
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Generic VWAP Final Balance: {final_balance}")

    trade_signals = trader_bots.VWAP_bot(
        ohlcv_df = ohlcv_df_test, 
        window = 12
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised VWAP Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic VWAP", "Optimised VWAP"], "Effect of GA Parameter Optimisation on VWAP Bot Performance")


### Stochastic Oscillator ###

    print("Stochastic Oscillator Trading Bot")
    trade_signals = trader_bots.stochastic_oscillator_bot(
        ohlcv_df = ohlcv_df_test, 
        oscillator_window = 14, 
        signal_window = 3, 
        overbought_threshold = 80, 
        oversold_threshold = 20
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Stochastic Oscillator Final Balance: {final_balance}")

    trade_signals = trader_bots.stochastic_oscillator_bot(
        ohlcv_df = ohlcv_df_test, 
        oscillator_window = 12, 
        signal_window = 5, 
        overbought_threshold = 70, 
        oversold_threshold = 30
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised Stochastic Oscillator Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic Stochastic Oscillator", "Optimised Stochastic Oscillator"], "Effect of GA Parameter Optimisation on Stochastic Oscillator Bot Performance")


### SAR ###

    print("SAR Trading Bot")
    trade_signals = trader_bots.SAR_bot(
        ohlcv_df = ohlcv_df_test, 
        step = 0.02,
        max_step = 0.2
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Generic SAR Final Balance: {final_balance}")

    trade_signals = trader_bots.SAR_bot(
        ohlcv_df = ohlcv_df_test, 
        step = 0.04,
        max_step = 0.4
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised SAR Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic SAR", "Optimised SAR"], "Effect of GA Parameter Optimisation on SAR Bot Performance")


### OBV Trend-Following ###

    print("OBV Trend-Following Trading Bot")
    trade_signals = trader_bots.OBV_trend_following_bot(
        ohlcv_df = ohlcv_df_test 
    ).generate_signals()
    final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    print(f"OBV Trend-Following Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results], ["Generic OBV Trend-Following"], "OBV Trend-Following Bot Performance")


### OBV Trend-Reversal ###

    print("OBV Trend-Reversal Trading Bot")
    trade_signals = trader_bots.OBV_trend_reversal_bot(
        ohlcv_df = ohlcv_df_test 
    ).generate_signals()
    final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    print(f"OBV Trend-Reversal Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results], ["Generic OBV Trend-Reversal"], "OBV Trend-Reversal Bot Performance")


### ROC ###

    print("ROC Trading Bot")
    trade_signals = trader_bots.ROC_bot(
        ohlcv_df = ohlcv_df_test,
        window = 12,
        buy_threshold = 5,
        sell_threshold = -5
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Generic ROC Final Balance: {final_balance}")

    trade_signals = trader_bots.ROC_bot(
        ohlcv_df = ohlcv_df_test,
        window = 8,
        buy_threshold = 10,
        sell_threshold = -10
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised ROC Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic ROC", "Optimised ROC"], "Effect of GA Parameter Optimisation on ROC Bot Performance")


### Awesome Oscillator ###

    print("Awesome Oscillator Trading Bot")
    trade_signals = trader_bots.Awesome_Oscillator_Bot(
        ohlcv_df = ohlcv_df_test,
        window1 = 5,
        window2 = 34
    ).generate_signals()
    final_balance, trade_results1 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Generic Awesome Oscillator Final Balance: {final_balance}")

    trade_signals = trader_bots.Awesome_Oscillator_Bot(
        ohlcv_df = ohlcv_df_test,
        window1 = 10,
        window2 = 23
    ).generate_signals()
    final_balance, trade_results2 = utils.execute_trades(trade_signals, fee_percentage)
    print(f"Optimised Awesome Oscillator Final Balance: {final_balance}\n")
    utils.plot_trading_simulation([trade_results1, trade_results2], ["Generic ROC", "Optimised ROC"], "Effect of GA Parameter Optimisation on ROC Bot Performance")



if __name__ == "__main__":

    fee_percentage = 0.02

    main(fee_percentage)
