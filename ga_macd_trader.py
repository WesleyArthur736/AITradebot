import random
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import ta
from ta.trend import MACD
from typing import List
import numpy as np

from ypstruct import structure


class TraderAgent:

    def __init__(self, window_slow, window_fast, ohclv_data):
        self.ohclv_data = ohclv_data
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.position = {"BTC": 0, "AUD": 100.0}
        self.net_worth = 100.0
        self.pnls = []

    def generate_signals(self, ohlcv):

        # macd_indicator = MACD(ohlcv["Mid Price"])
        macd_indicator = MACD(
            ohlcv["Mid Price"],
            window_slow = self.window_slow,
            window_fast = self.window_fast
        )

        # TODO rename this first signals var
        signals = np.where(macd_indicator.macd() > macd_indicator.macd_signal(), 1.0, 0.0)
        signals = np.diff(signals)
        signals = np.pad(signals, (1,), 'constant', constant_values=0)

        return signals

    def exec_trades(self):

        signals = self.generate_signals(self.ohclv_data)

        order = 0

        for idx, row in self.ohclv_data.iterrows():
            if order == 1:
                price = row["open"]
                self.position["BTC"] = self.position["AUD"] / price
                self.position["AUD"] = 0
                order = 0
            elif order == -1:
                price = row["open"]
                self.position["AUD"] = self.position["BTC"] * price
                self.position["BTC"] = 0
                order = 0
            signal = signals[idx]
            # buy
            if signal == 1.0:
                order = 1
            # sell
            elif signal == -1.0:
                order = -1

            # self.net_worth = self.position["BTC"] * row["Close"] + self.position["AUD"]
            self.net_worth = self.position["BTC"] * row["close"] + self.position["AUD"]
            self.pnls.append(self.net_worth)

        return self.net_worth

def fitness(trader_agent):
    # Implement a fitness function that evaluates the performance of the trader agent
    return trader_agent.exec_trades()

def crossover(parent1, parent2, ohclv_data):
    # Implement crossover operation that creates a new child by combining the genes of the two parents
    child = TraderAgent(parent1.window_slow, parent2.window_fast, ohclv_data)

    return child

def mutate(trader_agent, mutation_rate):
    # Implement mutation operation that randomly modifies the genes of the trader agent with a certain probability
    if random.random() < mutation_rate:
        trader_agent.window_slow = random.randint(1, 100)
    if random.random() < mutation_rate:
        trader_agent.window_fast = random.randint(1, 100)

    # if random.random() < mutation_rate:
    #     trader_agent.window_slow += random.randint(-10, 10)
    # if random.random() < mutation_rate:
    #     trader_agent.window_fast += random.randint(-10, 10)

    return trader_agent

def run_genetic_algorithm(ohclv_data, population_size, mutation_rate, num_generations):

    # Generate an initial population of trader agents with random parameters
    population = [
        TraderAgent(
            window_slow = random.randint(1, 100), 
            window_fast = random.randint(1, 100), 
            ohclv_data = ohclv_data
        ) for _ in range(population_size)
    ]
    
    for i in range(num_generations):

        print(f"\ngeneration: {i}")

        # Evaluate the fitness of each trader agent
        fitness_scores = [fitness(trader_agent) for trader_agent in population]
        
        # Select the top-performing trader agents to be parents of the next generation
        parents = [population[index] for index in sorted(range(len(fitness_scores)), key = lambda i: fitness_scores[i])[-2:]]
        
        # Create a new population by crossing over and mutating the parents
        new_population = [crossover(parents[0], parents[1], ohclv_data) for _ in range(population_size)]

        new_population = [mutate(trader_agent, mutation_rate) for trader_agent in new_population]
        
        # Replace the old population with the new one
        population = new_population
    
    # Return the best-performing trader agent
    fitness_scores = [fitness(trader_agent) for trader_agent in population]
    best_index = max(range(len(fitness_scores)), key = lambda i: fitness_scores[i])

    return population[best_index]

if __name__ == '__main__':
    # Define the parameters for the genetic algorithm
    population_size = 10
    mutation_rate = 0.05
    num_generations = 10

    # instantiate an exchange
    exchange = ccxt.kraken()

    # ohlcv data for BTC/AUD
    ohlcv_btc_aud = exchange.fetch_ohlcv("BTC/AUD", timeframe = "1d", limit = 720)

    # cast the ohlcv_btc_aud data to a pandas dataframe
    btc_aud_candles_ohclv = pd.DataFrame(
        ohlcv_btc_aud, 
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    # calculate based on the mod price
    btc_aud_candles_ohclv["Mid Price"] = 0.5 * btc_aud_candles_ohclv["high"] + 0.5 * btc_aud_candles_ohclv["close"]

    start = 0
    end = 500

    ohclv_data = btc_aud_candles_ohclv.loc[start:(end - 1)].reset_index(drop = False)

    # Run the genetic algorithm
    # best_trader_agent = run_genetic_algorithm(population_size, mutation_rate, num_generations)
    best_trader_agent = run_genetic_algorithm(ohclv_data, population_size, mutation_rate, num_generations)
    
    # Print the parameters of the best-performing trader agent
    print("\nBest trader agent:")
    print("Window slow:", best_trader_agent.window_slow)
    print("Window fast:", best_trader_agent.window_fast)