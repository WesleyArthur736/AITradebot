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

import utils
import trader_bots


class GeneticAlgorithmOptimizer(object):

    def __init__(self, ohlcv_df, trader_agent, trade_signals, fee_percentage, population_size, mutation_rate, num_generations):
        self.trader_agent = trader_agent
        self.trader_agent_params = trader_agent.params

        self.trade_signals = trade_signals
        self.fee_percentage = fee_percentage
        self.bot_type = trader_agent.bot_type

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations

        self.ohlcv_df = ohlcv_df

    def fitness(self, trader_agent, trade_signals, fee_percentage):
        # Implement a fitness function that evaluates the performance of the trader agent
        # aud_balance, _ = trader_agent.execute_trades(trade_signals, fee_percentage)
        aud_balance, _ = utils.execute_trades(trade_signals, fee_percentage)
        return aud_balance

    # def uniform_crossover(self, parent1, parent2):
    #     # weight the uniform_crossover for each param value
    #     # Implement uniform_crossover operation that creates a new child by combining the genes of the two parents
    #     child = MACD_Trading_Bot(parent1.SLOW_WINDOW, parent2.FAST_WINDOW, (parent1.SIGNAL_WINDOW + parent1.SIGNAL_WINDOW)/2)

    #     return child

    # def uniform_crossover(self, parent1, parent2):
    #     bot_type = parent1.bot_type
    #     for parent_1_param in parent1.params:
    #         for parent_2_param in parent2.params:
    #             child_params =

    #     return child

    # def uniform_crossover(self, parent1, parent2):
    #     # Create a new child bot with the same parameters as the parents
    #     child_params = [None] * len(parent1.params)
    #     for i in range(len(parent1.params)):
    #         if random.random() < 0.5:
    #             child_params[i] = parent1.params[i]
    #         else:
    #             child_params[i] = parent2.params[i]
    #     child_bot = Bot(child_params)
    #     # Other bot uniform_crossover code here
    #     return child_bot

    def uniform_crossover(self, parent1, parent2):
        """
        Implements uniform uniform_crossover. 
        parent1 and parent2 must be of the same type.
        """

        child_params = []

        for i in range(len(parent1.params)):
            if random.random() < 0.5:
                child_params.append(parent1.params[i])
            else:
                child_params.append(parent2.params[i])

        child_bot = type(parent1)(parent1.ohlcv_df, *child_params)

        return child_bot

    def mutate(self, trader_agent, mutation_rate, param_variance = 0.1, delta = 10):
        """
        Randomly modifies the genes of the trader agent with a certain probability.
        This is achieved by simply adding Gaussian white noise to the float parameters of 
        the bot. The float noise has a mean of 0 and a standard deviation of 0.1.
        In the case of integer parameters, a random nunber in the range -delta to delta is 
        added to the parameter.
        """

        # Add some Gaussian noise to the parameters, with a variance determined my "param_variance"
        for bot_param in trader_agent.params:
            if random.random() < mutation_rate:
                if isinstance(bot_param, float):
                    bot_param += np.random.normal(0, param_variance)
                elif isinstance(bot_param, int):
                    bot_param += np.random.randint(-delta, delta)
                else:
                    # sometimes I got this branch of the logic to run, which was a bit spooky
                    print(f"\n\n\nbot_param: {bot_param}")
                    print(f"type(bot_param): {type(bot_param)}\n\n\n")

        return trader_agent

    def run_genetic_algorithm(self):

        # need to pass the parameter values to optimise and then a list of acceptable bounds on these params

        # Generate an initial population of trader agents with random parameters

        # this unpacks a generic list of parameter values into the right bot constructor:
        population = [
            type(self.trader_agent)(
                self.ohlcv_df, *self.trader_agent_params
            ) for _ in range(population_size)
        ]
        
        for i in range(num_generations):

            print(f"\ngeneration: {i}")

            # Evaluate the fitness of each trader agent
            fitness_scores = [self.fitness(trader_agent, trade_signals, fee_percentage) for trader_agent in population]
            
            # Select the top-performing trader agents to be parents of the next generation
            parents = [population[index] for index in sorted(range(len(fitness_scores)), key = lambda i: fitness_scores[i])[-2:]]
            
            # Create a new population by crossing over and mutating the parents
            new_population = [self.uniform_crossover(parents[0], parents[1]) for _ in range(population_size)]

            new_population = [self.mutate(trader_agent, mutation_rate) for trader_agent in new_population]
            
            # Replace the old population with the new one
            population = new_population
        
        # Return the best-performing trader agent
        fitness_scores = [self.fitness(trader_agent, trade_signals, fee_percentage) for trader_agent in population]

        best_index = max(range(len(fitness_scores)), key = lambda i: fitness_scores[i])

        return population[best_index]


if __name__ == "__main__":

    ohlcv_df = utils.get_daily_ohlcv_data()

    fee_percentage = 0.0
    population_size = 100
    mutation_rate = 0.05
    num_generations = 10

    window = 50
    num_standard_deviations = 1.5

    overbought_threshold = 11
    oversold_threshold = 21

    oscillator_window = 20
    signal_window = 3

    max_step = 5
    step = 2

    buy_threshold = 100
    sell_threshold = 200

    min_literals = 2
    max_literals = 5
    min_conjunctions = 2
    max_conjunctions = 10

    # instantiate a bot - in this case the stochastic oscillator
    stoch_osc_bot = trader_bots.stochastic_oscillator_bot(
        ohlcv_df = ohlcv_df,
        oscillator_window = oscillator_window,
        signal_window = signal_window,
        overbought_threshold = overbought_threshold,
        oversold_threshold = oversold_threshold
    )

    # generate the trading signals with the bot's technical indicator:
    trade_signals = stoch_osc_bot.generate_signals()

    # instantiate a GeneticAlgorithmOptimizer
    ga_optimizer = GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df,
        trader_agent = stoch_osc_bot,
        trade_signals = trade_signals,
        fee_percentage = fee_percentage,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    ### Run the Genetic Algorithm ###
    best_agent = ga_optimizer.run_genetic_algorithm()

    # generate the trading signals with the bot's technical indicator:
    best_trade_signals = best_agent.generate_signals()

    print(f"Best agent's Trade Signals:\n{best_trade_signals}")

    best_final_balance, best_trade_results = utils.execute_trades(best_trade_signals, fee_percentage)

    print(f"Best agent's Trade Results:\n{best_trade_results}")
    print(f"Best agent's Final Balance:\n{best_final_balance}")

    utils.plot_trading_simulation(best_trade_results, "Best")


# def run_macd_trader():

#     # get the raw ohclv data in the format we require
#     ohlcv_df = get_daily_ohlcv_data()

#     # instantiate a trading bot:
#     macd_trading_bot = MACD_Trading_Bot(
#         slow_window = 26, 
#         fast_window = 12, 
#         signal_window = 9
#     )

#     # specify the fee percentage
#     fee_percentage = 0.02

#     # The MACD histogram is computed from the daily close prices.
#     # close_prices = trade_signals["close"]
#     close_prices = ohlcv_df["close"]

#     # create the indicator
#     indicator = MACD(
#         close_prices, 
#         window_slow = macd_trading_bot.SLOW_WINDOW, 
#         window_fast = macd_trading_bot.FAST_WINDOW, 
#         window_sign = macd_trading_bot.SIGNAL_WINDOW
#     )

#     # generate the trade signals with the indicaator
#     trade_signals = macd_trading_bot.generate_signals(ohlcv_df, indicator, close_prices)

#     # execute the trades determined by the indicator:
#     final_balance, trade_results = macd_trading_bot.execute_trades(trade_signals, fee_percentage)

#     print(f"MACD trade_results:\n{trade_results}")
#     print(f"MACD final_balance:\n{final_balance}")

#     plot_trading_simulation(trade_results)


# if __name__ == '__main__':

#     # uncomment this to run the macd trader
#     run_macd_trader()

#     # # we need to adress the potential for overfitting
#     # # implement k-fold cross validation with time series applicability.

#     # # Define the parameters for the genetic algorithm
#     # population_size = 10
#     # mutation_rate = 0.05
#     # num_generations = 10
#     # fee_percentage = 0.02

#     # ohlcv_df = get_daily_ohlcv_data()

#     # macd_trading_bot = MACD_Trading_Bot(
#     #     slow_window = 26, 
#     #     fast_window = 12, 
#     #     signal_window = 9
#     # )

#     # trade_signals = macd_trading_bot.generate_signals(ohlcv_df)

#     # # best_trader_agent = run_genetic_algorithm(ohlcv_df, population_size, mutation_rate, num_generations)
#     # best_trader_agent = run_genetic_algorithm(population_size, mutation_rate, num_generations, trade_signals, fee_percentage)

#     # print(f"best_trader_agent.SLOW_WINDOW: {best_trader_agent.SLOW_WINDOW}")
#     # print(f"best_trader_agent.FAST_WINDOW: {best_trader_agent.FAST_WINDOW}")
#     # print(f"best_trader_agent.SIGNAL_WINDOW: {best_trader_agent.SIGNAL_WINDOW}")

#     # best_final_balance, best_trade_results = best_trader_agent.execute_trades(trade_signals, fee_percentage)

#     # # print(f"Best MACD trade_results:\n{best_trade_results}")
#     # print(f"Best MACD final_balance:\n{best_final_balance}")

#     # plot_trading_simulation(best_trade_results)





# class MACD_Trading_Bot:
    
#     def __init__(self, slow_window, fast_window, signal_window):
#         self.SLOW_WINDOW = slow_window
#         self.FAST_WINDOW = fast_window
#         self.SIGNAL_WINDOW = signal_window

#     def execute_trades(self, trade_signals, fee_percentage):
#         """ Executes all of the identified trade signals sequentially.
#             Ensures the final holdings are in AUD.
#             Returns the trading account's final balance in AUD.
#         """
#         trade_results = trade_signals.copy()
#         trade_results["portfolio_value"] = 0

#         aud_balance = 100.00
#         btc_balance = 0.00

#         last_trade = "sell"

#         # For each day:
#         for index, row in trade_results.iterrows():
#             buy_signal = row["buy_signal"]
#             sell_signal = row["sell_signal"]
#             next_day_open_price = row["next_day_open"]
        
#             # Records daily portfolio value in AUD at market close.
#             if last_trade == "buy": 
#                 trade_results.at[index, "portfolio_value"] = btc_balance * row["close"]
#             elif last_trade == "sell":
#                 trade_results.at[index, "portfolio_value"] = aud_balance

#             # Executes trade at following day's open price if today's data results in trade signal.
#             if buy_signal == True and last_trade == "sell":  # Buy signal
#                 # Converts all AUD to BTC using the next day's open price and applies percentage fee.
#                 btc_balance = aud_balance / next_day_open_price * (1 - fee_percentage)
#                 aud_balance = 0
#                 last_trade = "buy"

#             elif sell_signal == True and last_trade == "buy":  # Sell signal
#                 # Converts all BTC to AUD using the next day's open price and applies percentage fee.
#                 aud_balance = btc_balance * next_day_open_price * (1 - fee_percentage)
#                 btc_balance = 0
#                 last_trade = "sell"

#         # Converts final holdings to AUD using final day's open price if final holdings are in BTC.
#         if last_trade == "buy":
#             last_close_price = trade_results["next_day_open"].iloc[-1]
#             aud_balance = btc_balance * last_close_price * (1 - fee_percentage)
#             btc_balance = 0

#         return aud_balance, trade_results


#     def generate_signals(self, ohlcv_df, indicator, close_prices):
#         """ Computes the MACD histogram using the daily close prices. 
#             Identifies the buy/sell signals (changes in histogram sign).
#             Returns a DataFrame with all the required data for executing the trades.
#         """

#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = ohlcv_df.copy()  

#         # # Computes MACD histogram.
#         # macd_indicator = MACD(close_prices, window_slow=self.SLOW_WINDOW, window_fast=self.FAST_WINDOW, window_sign=self.SIGNAL_WINDOW)
#         trade_signals["MACD_histogram"] = indicator.macd_diff()    # Computes indicator values.

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iloc[1:].iterrows():
#             # Evaluates literals. 
#             MACD_histogram_was_negative = 0 > trade_signals.at[index - 1, "MACD_histogram"]
#             MACD_histpgram_was_positive = trade_signals.at[index - 1, "MACD_histogram"] > 0
#             MACD_histogram_now_negative = 0 > trade_signals.at[index, "MACD_histogram"] 
#             MACD_histogram_now_positive = trade_signals.at[index, "MACD_histogram"] > 0
            
#             # Evaluates DNF formulas to determine buy and sell signals. 
#             buy_signal = MACD_histogram_was_negative and MACD_histogram_now_positive
#             sell_signal = MACD_histpgram_was_positive and MACD_histogram_now_negative

#             # Records buy and sell signals. 
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops unwanted column from output dataframe.
#         trade_signals = trade_signals.drop(columns = ["MACD_histogram"])

#         return trade_signals

# def get_daily_ohlcv_data():
#     """ Fetches the most recent 720 days of OHLCV data on BTC/AUD from Kraken.
#         Converts data into a Pandas DataFrame with column titles.
#         Alters and returns the DataFrame for further analysis.
#     """
#     exchange = kraken()
#     ohlcv_data = exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
#     ohlcv_df = DataFrame(ohlcv_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
#     ohlcv_df["next_day_open"] = ohlcv_df["open"].shift(-1)     # Adds column for next day's open price.
#     ohlcv_df = ohlcv_df.iloc[:-1]    # Removes last day's data as the bot cannot trade the next day.

#     return ohlcv_df

# def plot_trading_simulation(trade_results):
#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Set the x-axis data (day of trading) and y-axis data (portfolio value in AUD at close)
#     x_data = trade_results.index
#     y_data = trade_results["portfolio_value"]

#     # Plot the data
#     ax.plot(x_data, y_data)

#     # Set the labels and title
#     ax.set_xlabel("Day of Trading")
#     ax.set_ylabel("Portfolio Value in AUD at Close")
#     ax.set_title("Trading Simulation Results")

#     # Display the plot
#     plt.show()