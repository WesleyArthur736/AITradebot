import bots 
import random

# Get ohlcv data from Kraken.
ohlcv_df = bots.get_daily_ohlcv_data()

# Initialises ensemble buy and sell signal columns. 
ohlcv_df["buy_signal"] = False
ohlcv_df["sell_signal"] = False

# Determine the trade signals from each simple trading strategy.
MACD_signals = bots.MACD_Trading_Bot(slow_window = 26, fast_window = 12, signal_window = 9).determine_MACD_signals(ohlcv_df)
Bollinger_Bands_signals = trade_signals = bots.Bollinger_Bands_Trading_Bot(window = 20, num_standard_deviations = 2.5).determine_BB_signals(ohlcv_df)
# Update these with the bots when done. 
RSI_signals = MACD_signals
Stochastic_Oscillator_signals = MACD_signals
VWAP_signals = MACD_signals

# Defines strategy names.
strategy_names = ["MACD", "Bollinger_Bands", "RSI", "Stochastic_Oscillator", "VWAP"]

# Creates dictionary of simple strategy signals.
indicator_strategies = {
    "MACD": MACD_signals, 
    "Bollinger_Bands": Bollinger_Bands_signals, 
    "RSI": RSI_signals, 
    "Stochastic_Oscillator": Stochastic_Oscillator_signals, 
    "VWAP": VWAP_signals
}


def construct_conjunction(strategy_names, trade_type):
    # Chooses the strategies used in the conjunction.
    number_of_strategies = len(strategy_names)
    number_of_strategies_included = random.randint(1, number_of_strategies)
    strategies_used = random.sample(strategy_names, number_of_strategies_included)

    # Constructs the conjunction by ANDing the signals from the selected strategies.
    buy_signals = []
    for strategy_name in strategies_used:
        buy_signal = f"{strategy_name}_signals.at[index, '{trade_type}_signal']"
        buy_signals.append(buy_signal)
    conjunction = " and ".join(buy_signals)
    
    return conjunction


def construct_dnf(trade_type):
    # Chooses how many conjunctions are used in the DNF.
    number_of_conjunctions = random.randint(1, 4)

    # Constructs the DNF by generating conjunctions and ORing them together.
    conjunctions = []
    for i in range(number_of_conjunctions):
            conjunction = construct_conjunction(strategy_names, trade_type)
            conjunctions.append(conjunction)
    dnf = " or ".join(conjunctions)
    
    return dnf

# Create random DNF expression for buy signal.
buy_dnf = construct_dnf(trade_type = "buy")

# Evaluate DNF expression for each day of data and save to dataframe.
for index, row in ohlcv_df.iterrows():
    buy_dnf_with_index = buy_dnf.replace("index", str(index))
    print(buy_dnf_with_index)
    buy_signal = eval(buy_dnf_with_index)
    ohlcv_df.at[index, "buy_signal"] = buy_signal 

# Create random DNF expression for sell signal.
sell_dnf = construct_dnf(trade_type = "sell")

# Evaluate DNF expression for each day of data and save to dataframe.
for index, row in ohlcv_df.iterrows(): 
    sell_dnf_with_index = sell_dnf.replace("index", str(index))
    print(sell_dnf_with_index)
    sell_signal = eval(sell_dnf_with_index)
    ohlcv_df.at[index, "sell_signal"] = sell_signal 

# print(ohlcv_df.to_string())


final_balance, trade_results = bots.execute_trades(ohlcv_df, 0.02)
# print("Ensemble Trade Results:")
# print(trade_results)
# print("Ensemble Final Balance:")
# print(final_balance)
# print()
bots.plot_trading_simulation(trade_results)


