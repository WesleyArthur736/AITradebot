import random
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import ta
from ta.trend import MACD
from typing import List
import numpy as np

from ccxt import kraken
from pandas import DataFrame, concat
from ta.volatility import BollingerBands




class MACD_Trading_Bot:
    
    def __init__(self, slow_window, fast_window, signal_window):
        self.SLOW_WINDOW = slow_window
        self.FAST_WINDOW = fast_window
        self.SIGNAL_WINDOW = signal_window

    def execute_trades(self, trade_signals, fee_percentage):
        """ Executes all of the identified trade signals sequentially.
            Ensures the final holdings are in AUD.
            Returns the trading account's final balance in AUD.
        """
        trade_results = trade_signals.copy()
        trade_results["portfolio_value"] = 0

        aud_balance = 100.00
        btc_balance = 0.00

        last_trade = "sell"

        # For each day:
        for index, row in trade_results.iterrows():
            buy_signal = row["buy_signal"]
            sell_signal = row["sell_signal"]
            next_day_open_price = row["next_day_open"]
        
            # Records daily portfolio value in AUD at market close.
            if last_trade == "buy": 
                trade_results.at[index, "portfolio_value"] = btc_balance * row["close"]
            elif last_trade == "sell":
                trade_results.at[index, "portfolio_value"] = aud_balance

            # Executes trade at following day's open price if today's data results in trade signal.
            if buy_signal == True and last_trade == "sell":  # Buy signal
                # Converts all AUD to BTC using the next day's open price and applies percentage fee.
                btc_balance = aud_balance / next_day_open_price * (1 - fee_percentage)
                aud_balance = 0
                last_trade = "buy"

            elif sell_signal == True and last_trade == "buy":  # Sell signal
                # Converts all BTC to AUD using the next day's open price and applies percentage fee.
                aud_balance = btc_balance * next_day_open_price * (1 - fee_percentage)
                btc_balance = 0
                last_trade = "sell"

        # Converts final holdings to AUD using final day's open price if final holdings are in BTC.
        if last_trade == "buy":
            last_close_price = trade_results["next_day_open"].iloc[-1]
            aud_balance = btc_balance * last_close_price * (1 - fee_percentage)
            btc_balance = 0

        return aud_balance, trade_results


    def determine_MACD_signals(self, ohlcv_df):
        """ Computes the MACD histogram using the daily close prices. 
            Identifies the buy/sell signals (changes in histogram sign).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = ohlcv_df.copy()  

        # The MACD histogram is computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes MACD histogram.
        macd_indicator = MACD(close_prices, window_slow=self.SLOW_WINDOW, window_fast=self.FAST_WINDOW, window_sign=self.SIGNAL_WINDOW)
        trade_signals["MACD_histogram"] = macd_indicator.macd_diff()    # Computes indicator values.

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals. 
            MACD_histogram_was_negative = 0 > trade_signals.at[index - 1, "MACD_histogram"]
            MACD_histpgram_was_positive = trade_signals.at[index - 1, "MACD_histogram"] > 0
            MACD_histogram_now_negative = 0 > trade_signals.at[index, "MACD_histogram"] 
            MACD_histogram_now_positive = trade_signals.at[index, "MACD_histogram"] > 0
            
            # Evaluates DNF formulas to determine buy and sell signals. 
            buy_signal = MACD_histogram_was_negative and MACD_histogram_now_positive
            sell_signal = MACD_histpgram_was_positive and MACD_histogram_now_negative

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns = ["MACD_histogram"])

        return trade_signals

def get_daily_ohlcv_data():
    """ Fetches the most recent 720 days of OHLCV data on BTC/AUD from Kraken.
        Converts data into a Pandas DataFrame with column titles.
        Alters and returns the DataFrame for further analysis.
    """
    exchange = kraken()
    ohlcv_data = exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
    ohlcv_df = DataFrame(ohlcv_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
    ohlcv_df["next_day_open"] = ohlcv_df["open"].shift(-1)     # Adds column for next day's open price.
    ohlcv_df = ohlcv_df.iloc[:-1]    # Removes last day's data as the bot cannot trade the next day.

    return ohlcv_df

def plot_trading_simulation(trade_results):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the x-axis data (day of trading) and y-axis data (portfolio value in AUD at close)
    x_data = trade_results.index
    y_data = trade_results["portfolio_value"]

    # Plot the data
    ax.plot(x_data, y_data)

    # Set the labels and title
    ax.set_xlabel("Day of Trading")
    ax.set_ylabel("Portfolio Value in AUD at Close")
    ax.set_title("Trading Simulation Results")

    # Display the plot
    plt.show()

def fitness(trader_agent, trade_signals, fee_percentage):
    # Implement a fitness function that evaluates the performance of the trader agent
    aud_balance, _ = trader_agent.execute_trades(trade_signals, fee_percentage)
    return aud_balance

def crossover(parent1, parent2):
    # Implement crossover operation that creates a new child by combining the genes of the two parents
    child = MACD_Trading_Bot(parent1.SLOW_WINDOW, parent2.FAST_WINDOW, (parent1.SIGNAL_WINDOW + parent1.SIGNAL_WINDOW)/2)

    return child

def mutate(trader_agent, mutation_rate):
    # Implement mutation operation that randomly modifies the genes of the trader agent with a certain probability
    if random.random() < mutation_rate:
        trader_agent.SLOW_WINDOW = random.randint(1, 100)
        trader_agent.FAST_WINDOW = random.randint(1, 100)
        trader_agent.SIGNAL_WINDOW = random.randint(1, 100)

    # if random.random() < mutation_rate:
    #     trader_agent.window_slow += random.randint(-10, 10)
    # if random.random() < mutation_rate:
    #     trader_agent.window_fast += random.randint(-10, 10)

    return trader_agent

def run_genetic_algorithm(population_size, mutation_rate, num_generations, trade_signals, fee_percentage):

    # Generate an initial population of trader agents with random parameters
    population = [
        MACD_Trading_Bot(
            slow_window = random.randint(1, 100), 
            fast_window = random.randint(1, 100), 
            signal_window = random.randint(1, 100)
        ) for _ in range(population_size)
    ]
    
    for i in range(num_generations):

        print(f"\ngeneration: {i}")

        # Evaluate the fitness of each trader agent
        fitness_scores = [fitness(trader_agent, trade_signals, fee_percentage) for trader_agent in population]
        
        # Select the top-performing trader agents to be parents of the next generation
        parents = [population[index] for index in sorted(range(len(fitness_scores)), key = lambda i: fitness_scores[i])[-2:]]
        
        # Create a new population by crossing over and mutating the parents
        new_population = [crossover(parents[0], parents[1]) for _ in range(population_size)]

        new_population = [mutate(trader_agent, mutation_rate) for trader_agent in new_population]
        
        # Replace the old population with the new one
        population = new_population
    
    # Return the best-performing trader agent
    fitness_scores = [fitness(trader_agent, trade_signals, fee_percentage) for trader_agent in population]
    best_index = max(range(len(fitness_scores)), key = lambda i: fitness_scores[i])

    return population[best_index]

def run_macd_trader():

    ohlcv_df = get_daily_ohlcv_data()

    macd_trading_bot = MACD_Trading_Bot(
        slow_window = 26, 
        fast_window = 12, 
        signal_window = 9
    )

    trade_signals = macd_trading_bot.determine_MACD_signals(ohlcv_df)

    final_balance, trade_results = macd_trading_bot.execute_trades(trade_signals, 0.00)

    print(f"MACD trade_results:\n{trade_results}")
    print(f"MACD final_balance:\n{final_balance}")

    plot_trading_simulation(trade_results)

# def run_bollinger_band_trader():
#     print("Bollinger Bands Trading Bot")
#     trade_signals = Bollinger_Bands_Trading_Bot(window = 20, num_standard_deviations = 2.5).determine_BB_signals(ohlcv_df)
#     print()
#     print("Bollinger Bands Trade Signals:")
#     print(trade_signals)
#     print()
#     final_balance, trade_results = execute_trades(trade_signals, 0.00)
#     print("Bollinger Bands Trade Signals:")
#     print(trade_results)
#     print("Bollinger Bands Final Balance:")
#     print(final_balance)
#     print()
#     plot_trading_simulation(trade_results)

if __name__ == '__main__':

    # # uncomment this to run the macd trader
    # run_macd_trader()

    # Define the parameters for the genetic algorithm
    population_size = 50
    mutation_rate = 0.05
    num_generations = 10
    fee_percentage = 0.0

    ohlcv_df = get_daily_ohlcv_data()

    macd_trading_bot = MACD_Trading_Bot(
        slow_window = 26, 
        fast_window = 12, 
        signal_window = 9
    )

    trade_signals = macd_trading_bot.determine_MACD_signals(ohlcv_df)

    # best_trader_agent = run_genetic_algorithm(ohlcv_df, population_size, mutation_rate, num_generations)
    best_trader_agent = run_genetic_algorithm(population_size, mutation_rate, num_generations, trade_signals, fee_percentage)

    print(f"best_trader_agent.SLOW_WINDOW: {best_trader_agent.SLOW_WINDOW}")
    print(f"best_trader_agent.FAST_WINDOW: {best_trader_agent.FAST_WINDOW}")
    print(f"best_trader_agent.SIGNAL_WINDOW: {best_trader_agent.SIGNAL_WINDOW}")

    best_final_balance, best_trade_results = best_trader_agent.execute_trades(trade_signals, 0.00)

    # print(f"Best MACD trade_results:\n{best_trade_results}")
    print(f"Best MACD final_balance:\n{best_final_balance}")

    plot_trading_simulation(best_trade_results)