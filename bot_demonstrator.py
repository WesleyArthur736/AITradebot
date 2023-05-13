import trader_bots 
import utils
import numpy as np

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

    # print("SAR Trading Bot")
    # trade_signals = trader_bots.SAR_bot(
    #     ohlcv_df = ohlcv_df, 
    #     step = 0.02,
    #     max_step = 0.2
    # ).generate_signals()
    # print()
    # print("SAR Trade Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("SAR Trade Results:")
    # print(trade_results)
    # print("SAR Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "SAR")

    # print("OBV Trend-Following Trading Bot")
    # trade_signals = trader_bots.OBV_trend_following_bot(
    #     ohlcv_df = ohlcv_df 
    # ).generate_signals()
    # print()
    # print("OBV Trend-Following Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("OBV Trend-Following Results:")
    # print(trade_results)
    # print("OBV Trend-Following Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "OBV Trend-Following")

    # print("OBV Trend-Reversal Trading Bot")
    # trade_signals = trader_bots.OBV_trend_reversal_bot(
    #     ohlcv_df = ohlcv_df 
    # ).generate_signals()
    # print()
    # print("OBV Trend-Reversal Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("OBV Trend-Reversal Results:")
    # print(trade_results)
    # print("OBV Trend-Reversal Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "OBV Trend-Reversal")

    # print("ROC Trading Bot")
    # trade_signals = trader_bots.ROC_bot(
    #     ohlcv_df = ohlcv_df,
    #     window = 12,
    #     buy_threshold = 5,
    #     sell_threshold = -5
    # ).generate_signals()
    # print()
    # print("ROC Signals:")
    # print(trade_signals)
    # print()
    # final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
    # print("ROC Results:")
    # print(trade_results)
    # print("ROC Final Balance:")
    # print(final_balance)
    # print()
    # utils.plot_trading_simulation(trade_results, "ROC")





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
    np.random.seed(42)
    MACD_sigp=np.random.randint(5,10)
    MACD_low=np.random.randint(11,20)
    MACD_high=np.random.randint(21,50)
    
    BB_win =np.random.randint(10,30)
    BBstd = np.random.uniform(0.5, 2.5)
    
    RSI_win= np.random.randint(10,30)
    Os_th= np.random.randint(10,50)
    Ob_th= np.random.randint(51,90)
    
    vwap_win = np.random.randint(10,30)
   
    StOsc_sm = np.random.randint(1,6)
    StOsc_win =np.random.randint(7,30)
    StOsc_Os = np.random.randint(1,49)
    StOsc_Ob = np.random.randint(50,99)
    
    psar_step = np.random.uniform(0.001, 0.03)
    psar_mstep = np.random.uniform(psar_step + 0.01,0.3)
    
    ROC_win = np.random.randint(5,20)
    ROC_bth = np.random.randint(1,20)
    ROC_sth = np.random.randint(-20,-1)
    
    AO_win1 = np.random.randint(20,40)
    AO_win2 = np.random.randint(1,10)
    
    IC_win1= np.random.randint(1,15)
    IC_win2= np.random.randint(16,35)
    IC_win3= np.random.randint(36,70)
    
    MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': MACD_sigp, 'fast_window': MACD_low, 'signal_window': MACD_high}
    Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': BB_win, 'num_standard_deviations': BBstd}
    RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': Ob_th, 'oversold_threshold': Os_th, 'window': RSI_win}
    VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': vwap_win}
    Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': StOsc_sm, 'signal_window': StOsc_win, 'overbought_threshold': StOsc_Ob, 'oversold_threshold': StOsc_Ob}
    SAR_parameters = {'bot_name': 'SAR_bot', 'step': psar_step, 'max_step': psar_mstep}
    OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
    OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
    ROC_parameters = {'bot_name': 'ROC_bot', 'window': ROC_win, 'buy_threshold': ROC_bth, 'sell_threshold': ROC_sth}
    AO_parameters = {'bot_name': 'AO_bot', 'window_long':AO_win1, 'window_short':AO_win2}
    IC_parameters = {'bot_name': 'IC_bot', 'window_low':IC_win1, 'window_medium':IC_win2, 'window_high':IC_win3 }
    
    all_parameters = [
        MACD_parameters, 
        Bollinger_Bands_parameters, 
        RSI_parameters, 
        VWAP_parameters, 
        Stochastic_Oscillator_parameters,
        OBV_trend_following_parameters,
        SAR_parameters,
        OBV_trend_reversal_parameters,
        ROC_parameters,
        AO_parameters,
        IC_parameters
        ]

    main(fee_percentage)
