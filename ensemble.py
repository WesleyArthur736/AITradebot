import functions 
import random

# Get ohlcv data from Kraken.
ohlcv_df = functions.get_daily_ohlcv_data()

# Initialises ensemble buy and sell signal columns. 
ohlcv_df["buy_signal"] = False
ohlcv_df["sell_signal"] = False

# Determine the trade signals from each simple trading strategy.
MACD_signals = functions.Bots(ohlcv_df).MACD_bot(slow_window = 26, fast_window = 12, signal_window = 9)
Bollinger_Bands_signals = functions.Bots(ohlcv_df).bollinger_bands_bot(window = 20, num_standard_deviations = 2.5)
RSI_signals = functions.Bots(ohlcv_df).RSI_bot(overbought = 70, oversold = 30, window = 14)
VWAP_signals = functions.Bots(ohlcv_df).VWAP_bot(window = 20)
# Update this with the bots when done. 
Stochastic_Oscillator_signals = MACD_signals


# Defines strategy names.
strategy_names = ["MACD", "Bollinger_Bands", "RSI", "VWAP", "Stochastic_Oscillator"]

# Creates dictionary of simple strategy signals.
indicator_strategies = {
    "MACD": MACD_signals, 
    "Bollinger_Bands": Bollinger_Bands_signals, 
    "RSI": RSI_signals, 
    "VWAP": VWAP_signals,
    "Stochastic_Oscillator": Stochastic_Oscillator_signals
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
print(buy_dnf)

# Evaluate DNF expression for each day of data and save to dataframe.
for index, row in ohlcv_df.iterrows():
    buy_dnf_with_index = buy_dnf.replace("index", str(index))
    buy_signal = eval(buy_dnf_with_index)
    ohlcv_df.at[index, "buy_signal"] = buy_signal 

# Create random DNF expression for sell signal.
sell_dnf = construct_dnf(trade_type = "sell")
print(sell_dnf)

# Evaluate DNF expression for each day of data and save to dataframe.
for index, row in ohlcv_df.iterrows(): 
    sell_dnf_with_index = sell_dnf.replace("index", str(index))
    sell_signal = eval(sell_dnf_with_index)
    ohlcv_df.at[index, "sell_signal"] = sell_signal 



final_balance, trade_results = functions.execute_trades(ohlcv_df, 0.02)
functions.plot_trading_simulation(trade_results)


