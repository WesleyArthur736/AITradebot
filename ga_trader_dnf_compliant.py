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

import ccxt
from ta.momentum import StochasticOscillator
import copy


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
# lst is list of parameters for the genetic algorithm
def run_genetic_algorithm(lst):#(population_size, mutation_rate, num_generations, trade_signals, fee_percentage):
    population_size, mutation_rate, num_generations, trade_signals, fee_percentage, tradebot = lst[0], lst[1], lst[2], lst[3], lst[4], lst[5]
    MONEY=100
    # Generate an initial population of trader agents with random parameters
    match tradebot:
        case "MACD":
            population = [
                MACD_Trading_Bot(
                    slow_window = random.randint(1, 100), 
                    fast_window = random.randint(1, 100), 
                    signal_window = random.randint(1, 100)
                ) for _ in range(population_size)
            ]
        case "StochOscill":
            population=[]
            for inx in range(population_size):
                STOCH_SIGNAL_WINDOW = random.randint(1, 10)
                STOCH_OSCILL_WINDOW = random.randint(11, 100)
                bitcoin_data, sosc_data, so_signal = set_up_objects([STOCH_OSCILL_WINDOW, STOCH_SIGNAL_WINDOW])
                population.append(stochasticOscillator(bitcoin_data, sosc_data, so_signal, MONEY))
                                
    
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
    params=[population_size, mutation_rate, num_generations, trade_signals, fee_percentage, "MACD"]
    best_trader_agent = run_genetic_algorithm(params)

    print(f"best_trader_agent.SLOW_WINDOW: {best_trader_agent.SLOW_WINDOW}")
    print(f"best_trader_agent.FAST_WINDOW: {best_trader_agent.FAST_WINDOW}")
    print(f"best_trader_agent.SIGNAL_WINDOW: {best_trader_agent.SIGNAL_WINDOW}")

    best_final_balance, best_trade_results = best_trader_agent.execute_trades(trade_signals, 0.00)

    # print(f"Best MACD trade_results:\n{best_trade_results}")
    print(f"Best MACD final_balance:\n{best_final_balance}")

    plot_trading_simulation(best_trade_results)
    

def main1():
    STOCH_OSCILL_WINDOW = 20
    STOCH_SIGNAL_WINDOW = 3
    MONEY = 100
    
    bitcoin_data, sosc_data, so_signal = set_up_objects([STOCH_OSCILL_WINDOW, STOCH_SIGNAL_WINDOW])
    soscill_bot = stochasticOscillator(bitcoin_data, sosc_data, so_signal, MONEY)
    
    #sell_triggers, buy_triggers 
    #orders = soscill_bot.formulate_orders(sell_triggers, buy_triggers)
    ndo, sell_signal, buy_signal = soscill_bot.execute()
    bitcoin_data= bitcoin_data[0:719]
    
    buy_signal.reset_index(drop=True,inplace=True)
    sell_signal.reset_index(drop=True,inplace=True)
    ndo.reset_index(drop=True,inplace=True)
    
    bot_data = pd.concat([bitcoin_data, ndo, buy_signal, sell_signal], axis = 1)
    print(bot_data)
    
    return bot_data
    
    
    
    

def set_up_objects(lst):
    kraken_exchange = ccxt.kraken()
    kraken_market = kraken_exchange.load_markets()
    
    bitcoin_data = kraken_exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
    bitcoin_indicators = pd.DataFrame(bitcoin_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
    
    stochOsc = StochasticOscillator(bitcoin_indicators['close'],bitcoin_indicators['high'],bitcoin_indicators['low'],window=lst[0],smooth_window=lst[1])
    sosc_data = stochOsc.stoch()
    so_signal = stochOsc.stoch_signal()
    
    return bitcoin_indicators, sosc_data, so_signal

class stochasticOscillator:
    def __init__(self,bitcoin_data, sosc_data, so_signal, MONEY ):
        self.bitcoin_Data =bitcoin_data
        self.stochOsc= sosc_data
        self.so_signal= so_signal
        self.cash= MONEY
        self.coins = 0
        
    def get_triggers(self):
        
        sell_triggers = []
        buy_triggers = []
        sosc= self.stochOsc.to_numpy()
        so_s= self.so_signal.to_numpy()
        
        # prepare arrays to give point in ranges of significance to the Stochastic Oscillator index
        so_overbought = np.argwhere(sosc>80)
        so_oversold = np.argwhere(sosc<20)
        
        so_crossover_sig = np.argwhere(np.greater(so_s , sosc))
        crossover_back = np.argwhere(np.greater(sosc , so_s))
        crossover_booleans_osc = np.in1d(crossover_back, so_crossover_sig + 1 )
        crossover_booleans_sig= np.in1d(so_crossover_sig,crossover_back +1)
        
        # Determine the triggers for selling
        # Output a list of selling triggers
        for inx in so_overbought: 
            if np.any(crossover_back == inx) == True: 
                idx_crossover = np.where(crossover_back == inx) #suspect contradiction
                print(idx_crossover[0], inx)
                
                if crossover_booleans_osc[idx_crossover[0]] == True:
                    print("i made it to here")
                    if so_s[inx] < sosc[inx]: 
                        #print("loading variable")
                        sell_triggers.append((inx[0], "-1"))
        print(sell_triggers)
        
        # Determine the triggers for buying
        for inx in so_oversold: 
            if np.any(so_crossover_sig == inx) == True:
                idx_crossover = np.where(so_crossover_sig == inx)
                
                if crossover_booleans_sig[idx_crossover[0]] == True:
                    if so_s[inx] > sosc[inx]:
                        buy_triggers.append((inx[0], 1))
                    
        print (buy_triggers)      
        
        return sell_triggers,buy_triggers 
    
        
    
    def formulate_orders(self, sell_triggers,buy_triggers):
        NO_DAYS_TRADE = 719
        orders = np.zeros((NO_DAYS_TRADE))
        for tup in buy_triggers:
            if tup[0] < NO_DAYS_TRADE:
                
                print("tup",tup[0])
            
                orders[tup[0]]=tup[1]
                
        for tup in sell_triggers:
            if tup[0] < NO_DAYS_TRADE:
                orders[tup[0]]=tup[1]
        
        return pd.Series(orders.astype(int), name ="trade_signal")
    
    def execute(self):
        STOCH_OSCILL_OVERBOUGHT = 80
        STOCH_OSCILL_OVERSOLD = 20
        portfolio_value=[]
        sell_signal = []
        buy_signal = []
        sosc= self.stochOsc.to_numpy()
        so_s= self.so_signal.to_numpy()
        
        ndo = self.bitcoin_Data['open'][1:720]
        print(ndo)
        
        
        # but Trigger
        for day in range(len(ndo)):
            crossover=[sosc[day] > so_s[day], sosc[day+1] > so_s[day+1]]

            buy_trigger = sosc[day] < STOCH_OSCILL_OVERSOLD and (crossover[0] == False and crossover[1] == True )
            sell_trigger = sosc[day] > STOCH_OSCILL_OVERBOUGHT and (crossover[0] == True and crossover[1] == False)

            #buy_trigger
            if buy_trigger == True and self.cash > 0:
                available_funds = self.cash - (self.cash/50)
                self.coins = float(self.coins + (available_funds/ ndo.iloc[day]))
                self.cash = 0 
            # sell trigger  
            elif sell_trigger == True and self.coins > 0:
                potential_funds= ndo.iloc[day] * self.coins
                funds_gained = potential_funds - (potential_funds/50)
                self.cash = float(self.cash + funds_gained)
                self.coins = 0
            portfolio_value.append(self.cash)
            buy_signal.append(buy_trigger)
            sell_signal.append(sell_trigger)
                
        if self.coins > 0:
            self.cash = float(self.cash + (ndo.iloc[day] * self.coins))
            self.coins = 0   
            portfolio_value[len(portfolio_value)-1] = self.cash
        print(self.cash,self.coins)
        
        
        
        return pd.Series(ndo, name = "next_day_open"), pd.Series(buy_signal, name="buy_signal"), pd.Series(sell_signal, name="sell_signal")
    
        
main1()