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

from sklearn.model_selection import train_test_split, KFold


class GeneticAlgorithmOptimizer(object):

    # def __init__(self, ohlcv_df, trader_agent, trade_signals, fee_percentage, population_size, mutation_rate, num_generations):
    def __init__(self, ohlcv_df, trader_agent, fee_percentage, population_size, mutation_rate, num_generations):
        self.trader_agent = trader_agent
        self.trader_agent_params = trader_agent.params
        
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


    # def run_genetic_algorithm(self, n_elite, tournament_size):
    def run_genetic_algorithm(self, population, n_elite, tournament_size):
        """
        Implements 'Elitism' and uses tournament selection for the parents
        """

        # # Generate an initial population of trader agents with random parameters
        # population = [
        #     type(self.trader_agent)(
        #         ohlcv_df = self.ohlcv_df, 
        #         *self.trader_agent_params
        #     ) for _ in range(self.population_size)
        # ] # THIS IS NOT CORRECT!!!!!

        # Generate an initial population of trader agents with random parameters
        # population = [
        #     type(self.trader_agent)(
        #         self.ohlcv_df, 
        #         *self.trader_agent_params
        #     ) for _ in range(self.population_size)
        # ] # THIS IS NOT CORRECT!!!!!

        for i in range(self.num_generations):

            print(f"\ngeneration: {i}")

            # Evaluate the fitness of each trader agent
            # fitness_scores = [self.fitness(trader_agent, self.trade_signals, self.fee_percentage) for trader_agent in population]
            fitness_scores = [self.fitness(trader_agent, trader_agent.generate_signals(), self.fee_percentage) for trader_agent in population]

            # Select the top-performing trader agents to be the elite members of the next generation
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:n_elite]
            elite_population = [population[i] for i in elite_indices]

            # Randomly select the rest of the parents for the next generation using tournament selection
            num_parents = self.population_size - n_elite
            parents = []
            for j in range(num_parents):
                # Select a random subset of the population to compete in the tournament
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament = [population[i] for i in tournament_indices]

                # Choose the best individual from the tournament as a parent
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                best_index = max(range(len(tournament_fitness)), key=lambda i: tournament_fitness[i])
                best_parent = tournament[best_index]

                parents.append(best_parent)

            # Create a new population by crossing over and mutating the parents
            new_population = []
            for i in range(self.population_size):
                if i < n_elite:
                    # Preserve elite members for the next generation
                    new_population.append(elite_population[i])
                else:
                    # Select two random parents for uniform crossover
                    parent1_index, parent2_index = random.sample(range(len(parents)), 2)
                    parent1 = parents[parent1_index]
                    parent2 = parents[parent2_index]

                    # Perform uniform crossover to create a new child bot
                    child_bot = self.uniform_crossover(parent1, parent2)

                    # Mutate the child bot with a certain probability
                    child_bot = self.mutate(child_bot, self.mutation_rate)

                    # Add the child bot to the new population
                    new_population.append(child_bot)

            # Replace the old population with the new one
            population = new_population

        # Return the best-performing trader agent
        # fitness_scores = [self.fitness(trader_agent, self.generate_signals(), self.fee_percentage) for trader_agent in population]
        fitness_scores = [self.fitness(trader_agent, trader_agent.generate_signals(), self.fee_percentage) for trader_agent in population]

        best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])

        return population[best_index]


class EnsembleGeneticAlgorithmOptimizer(object):

    def __init__(
            self, ohlcv_df, trader_agent, trade_signals, fee_percentage, 
            population_size, mutation_rate, num_generations, 
            number_of_disjuncts, number_of_conjuncts, all_strategies
        ):

        self.trader_agent = trader_agent
        self.trader_agent_params = trader_agent.params

        self.trade_signals = trade_signals
        self.fee_percentage = fee_percentage
        self.bot_type = trader_agent.bot_type

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations

        self.number_of_disjuncts = number_of_disjuncts
        self.number_of_conjuncts = number_of_conjuncts

        self.all_strategies = all_strategies

        self.ohlcv_df = ohlcv_df

    def fitness(self, trader_agent, trade_signals, fee_percentage):
        # Implement a fitness function that evaluates the performance of the trader agent
        aud_balance, _ = utils.execute_trades(trade_signals, fee_percentage)
        return aud_balance

    def ensemble_uniform_crossover(self, parent1, parent2):

        child_params = []

        # parent1.params = [parent1.number_of_disjuncts, parent1.strategies_to_use, parent1.buy_dnf, parent1.sell_dnf, parent1.number_of_conjuncts]
        # parent2.params = [parent2.number_of_disjuncts, parent2.strategies_to_use, parent2.buy_dnf, parent2.sell_dnf, parent2.number_of_conjuncts]
        for p_1_param, p_2_param in zip(parent1.params, parent2.params):
            if random.random() < 0.5:
                child_params.append(p_1_param)
            else:
                child_params.append(p_2_param)

        child_bot = type(self.trader_agent)(
            ohlcv_df = self.trader_agent.ohlcv_df,
            buy_dnf = child_params[2],
            sell_dnf = child_params[3],
            strategies_to_use = child_params[1],
            constituent_bot_parameters = self.trader_agent.constituent_bot_parameters,
            number_of_disjuncts = child_params[0],
            all_strategies = self.trader_agent.all_strategies,
            number_of_conjuncts = child_params[4]
        )

        return child_bot

    def ensemble_mutate(self, trader_agent, mutation_rate, num_disjuncts_mutate, all_strategies):

        # ensemble_bot.params = [self.number_of_disjuncts, self.strategies_to_use, self.buy_dnf, self.sell_dnf]

        trader_agent.params[2] = utils.mutate_dnf(
            trader_agent.params[2], 
            is_buy_dnf = True, 
            all_strategies = all_strategies,
            num_disjuncts_mutate = num_disjuncts_mutate
        )

        trader_agent.params[3] = utils.mutate_dnf(
            trader_agent.params[3], 
            is_buy_dnf = False, 
            all_strategies = all_strategies,
            num_disjuncts_mutate = num_disjuncts_mutate
        )

        return trader_agent

    def run_genetic_algorithm_ensemble(self, population, n_elite, tournament_size, num_disjuncts_mutate, all_strategies):

        # create a list of "trader_bot.ensemble_bot"'s, with randomly initialised parameter values
        # this is the population of solutions.
        # population = [
        #     type(self.trader_agent)(
        #         ohlcv_df = self.trader_agent.ohlcv_df,
        #         buy_dnf = utils.construct_dnf(
        #             trade_type = "buy", 
        #             number_of_disjuncts = random.randint(2, 5), 
        #             strategies_to_use = init_strategies_to_use,
        #             all_strategies = all_strategies,
        #             number_of_conjuncts = random.randint(1, 3)
        #         ),
        #         sell_dnf = utils.construct_dnf(
        #             trade_type = "sell", 
        #             number_of_disjuncts = random.randint(2, 5), 
        #             strategies_to_use = init_strategies_to_use,
        #             all_strategies = all_strategies,
        #             number_of_conjuncts = random.randint(1, 3)
        #         ),
        #         strategies_to_use = utils.select_initial_strats(all_strategies, number_of_conjuncts = random.randint(1, 3)),
        #         constituent_bot_parameters = self.trader_agent.constituent_bot_parameters,
        #         number_of_disjuncts = random.randint(2, 5),
        #         all_strategies = all_strategies,
        #         number_of_conjuncts = random.randint(1, 3)
        #     ) for i in range(self.population_size)
        # ]

        for i in range(self.num_generations):

            print(f"\ngeneration: {i}")

            # Evaluate the fitness of each trader agent
            fitness_scores = [self.fitness(trader_agent, trader_agent.trade_signals, self.fee_percentage) for trader_agent in population]

            for ensb_bot in population:
                # print(f"\nensb_bot.buy_dnf:\n{ensb_bot.buy_dnf}\n")
                # print(f"ensb_bot.sell_dnf:\n{ensb_bot.sell_dnf}\n")
                print(f"\nfitness_scores_1: {fitness_scores}\n")

            # Select the top-performing trader agents to be the elite members of the next generation
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:n_elite]
            elite_population = [population[i] for i in elite_indices]

            # Randomly select the rest of the parents for the next generation using tournament selection
            num_parents = self.population_size - n_elite
            parents = []
            for j in range(num_parents):
                # Select a random subset of the population to compete in the tournament

                # print(f"\nrandom.sample(list(range(len(population))), tournament_size): {random.sample(list(range(len(population))), tournament_size)}")
                # print(f"tournament_size: {tournament_size}\n")

                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament = [population[i] for i in tournament_indices]

                # Choose the best individual from the tournament as a parent
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                best_index = max(range(len(tournament_fitness)), key=lambda i: tournament_fitness[i])
                best_parent = tournament[best_index]

                parents.append(best_parent)

            # Create a new population by crossing over and mutating the parents
            new_population = []
            for i in range(self.population_size):
                if i < n_elite:
                    # Preserve elite members for the next generation
                    new_population.append(elite_population[i])
                else:
                    # Select two random parents for uniform crossover
                    parent1_index, parent2_index = random.sample(range(len(parents)), 2)
                    parent1 = parents[parent1_index]
                    parent2 = parents[parent2_index]

                    # Perform uniform crossover to create a new child bot
                    child_bot = self.ensemble_uniform_crossover(parent1, parent2)

                    # Mutate the child bot with a certain probability
                    child_bot = self.ensemble_mutate(child_bot, self.mutation_rate, num_disjuncts_mutate, all_strategies)

                    # Add the child bot to the new population
                    new_population.append(child_bot)


            # Replace the old population with the new one
            population = new_population

        # Return the best-performing trader agent
        fitness_scores = [self.fitness(trader_agent, trader_agent.trade_signals, self.fee_percentage) for trader_agent in population]

        for ensb_bot in population:
            # print(f"\nensb_bot.buy_dnf:\n{ensb_bot.buy_dnf}\n")
            # print(f"ensb_bot.sell_dnf:\n{ensb_bot.sell_dnf}\n")
            print(f"\nfitness_scores_1: {fitness_scores}\n")

        # all_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        # population = [population[i] for i in all_indices]
        # return population[0]

        best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])

        return population[best_index]


    def brute_force_search_ensemble(self, population):

        # Evaluate the fitness of each trader agent
        fitness_scores = [self.fitness(trader_agent, trader_agent.trade_signals, self.fee_percentage) for trader_agent in population]
        score_agent_pairs = []
        for fitness_score, trader_agent in zip(fitness_scores, population):
            score_agent_pairs.append([fitness_score, trader_agent])

        best_instance_so_far = max(score_agent_pairs, key = lambda x: x[0])

        for i in range(self.num_generations):

            print(f"\ngeneration: {i}")

            for trader_agent in population:
                trader_agent.trade_signals = trader_agent.generate_signals()

            # Evaluate the fitness of each trader agent
            fitness_scores = [self.fitness(trader_agent, trader_agent.trade_signals, self.fee_percentage) for trader_agent in population]
            score_agent_pairs = []
            for fitness_score, trader_agent in zip(fitness_scores, population):
                score_agent_pairs.append([fitness_score, trader_agent])

            contender = max(score_agent_pairs, key = lambda x: x[0])

            if contender[0] > best_instance_so_far[0]:
                best_instance_so_far = contender

        return best_instance_so_far[1]



# if __name__ == "__main__":

#     # ohlcv_df = utils.get_daily_ohlcv_data()
#     ohlcv_df_train = utils.get_daily_ohlcv_data()
#     # ohlcv_df_train, ohlcv_df_test = train_test_split(ohlcv_df, test_size = 0.2, shuffle = False)

#     fee_percentage = 0.02

#     population_size = 20
#     mutation_rate = 0.01
#     num_generations = 2
#     n_elite = 2
#     init_number_of_disjuncts = 5
#     init_number_of_conjuncts = 2

#     window = 50
#     num_standard_deviations = 1.5
#     overbought_threshold = 11
#     oversold_threshold = 21
#     oscillator_window = 20
#     signal_window = 3
#     max_step = 5
#     step = 2
#     buy_threshold = 100
#     sell_threshold = 200
#     min_literals = 2
#     max_literals = 5
#     min_conjunctions = 2
#     max_conjunctions = 10
#     slow_window = 26
#     fast_window = 12
#     signal_window = 9

#     # trader_agent = trader_bots.MACD_bot(
#     #     ohlcv_df = ohlcv_df_train,
#     #     slow_window = slow_window, 
#     #     fast_window = fast_window, 
#     #     signal_window = signal_window
#     # )

#     # trade_signals = trader_agent.generate_signals()

#     # # un-optimized bot
#     # final_balance, trade_results = utils.execute_trades(
#     #     trade_signals = trade_signals, 
#     #     fee_percentage = 0.0
#     # )

#     # ga_optimiser = GeneticAlgorithmOptimizer(
#     #     ohlcv_df = ohlcv_df_train,
#     #     trader_agent = trader_agent,
#     #     trade_signals = trade_signals,
#     #     fee_percentage = 0.0,
#     #     population_size = population_size,
#     #     mutation_rate = mutation_rate,
#     #     num_generations = num_generations
#     # )

#     # population = [
#     #     trader_bots.MACD_bot(
#     #         ohlcv_df = ohlcv_df_train,
#     #         slow_window = random.randint(1, 100),
#     #         fast_window = random.randint(1, 100),
#     #         signal_window = random.randint(1, 100),
#     #     ) for _ in range(population_size)
#     # ] 

#     # best_trader = ga_optimiser.run_genetic_algorithm(
#     #     population = population,
#     #     n_elite = 2, 
#     #     tournament_size = 5
#     # )

#     # # best_trade_signals = best_trader.trade_signals()
#     # best_trade_signals = best_trader.generate_signals()

#     # # optimized bot
#     # best_final_balance, best_trade_results = utils.execute_trades(
#     #     trade_signals = best_trade_signals, 
#     #     fee_percentage = 0.0
#     # )
#     # utils.plot_trading_simulation(trade_results, "Random MACD", color = "red")
#     # utils.plot_trading_simulation(best_trade_results, "Optimized MACD", color = "green")






#     MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
#     Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
#     RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
#     VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
#     Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14, 'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}
#     SAR_parameters = {'bot_name': 'SAR_bot', 'step': 0.02, 'max_step': 0.2}
#     OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
#     OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
#     ROC_parameters = {'bot_name': 'ROC_bot', 'window': 12, 'buy_threshold': 5, 'sell_threshold': -5}
#     Awesome_Osillator = {'bot_name': 'Awesome_Oscillator_Bot', 'window1': 5 , 'window2': 34}

#     constituent_bot_parameters = [ 
#         Bollinger_Bands_parameters, 
#         MACD_parameters,
#         RSI_parameters, 
#         VWAP_parameters, 
#         Stochastic_Oscillator_parameters,
#         OBV_trend_following_parameters,
#         SAR_parameters,
#         OBV_trend_reversal_parameters,
#         ROC_parameters,
#         Awesome_Osillator
#     ]

#     all_strategies = [
#         'MACD_bot', 'bollinger_bands_bot', 
#         'RSI_bot', 'VWAP_bot', 
#         'stochastic_oscillator_bot', 
#         'SAR_bot',
#         'OBV_trend_following_bot',
#         'OBV_trend_reversal_bot',
#         'ROC_bot', 'Awesome_Oscillator_Bot'
#     ]

#     # init strats randomly 
#     init_strategies_to_use = utils.select_initial_strats(all_strategies, init_number_of_conjuncts)

#     # construct initial buy_dnf
#     init_buy_dnf = utils.construct_dnf(
#         trade_type = "buy", 
#         number_of_disjuncts = init_number_of_disjuncts, 
#         strategies_to_use = init_strategies_to_use,
#         all_strategies = all_strategies,
#         number_of_conjuncts = init_number_of_conjuncts
#     )

#     # construct initial sell_dnf
#     init_sell_dnf = utils.construct_dnf(
#         trade_type = "sell", 
#         number_of_disjuncts = init_number_of_disjuncts, 
#         strategies_to_use = init_strategies_to_use,
#         all_strategies = all_strategies,
#         number_of_conjuncts = init_number_of_conjuncts
#     )


#     ########################################

#     trader_agent = trader_bots.ensemble_bot(
#         ohlcv_df = ohlcv_df_train,
#         buy_dnf = init_buy_dnf,
#         sell_dnf = init_sell_dnf,
#         strategies_to_use = init_strategies_to_use,
#         constituent_bot_parameters = constituent_bot_parameters,
#         number_of_disjuncts = init_number_of_disjuncts,
#         all_strategies = all_strategies,
#         number_of_conjuncts = init_number_of_conjuncts
#     )

#     trade_signals, _, _ = trader_agent.generate_signals()

#     # un-optimized bot
#     final_balance, trade_results = utils.execute_trades(
#         trade_signals = trade_signals, 
#         fee_percentage = 0.0
#     )

#     population = [
#         trader_bots.ensemble_bot(
#             ohlcv_df = ohlcv_df_train,
#             buy_dnf = utils.construct_dnf(
#                 trade_type = "buy", 
#                 number_of_disjuncts = random.randint(2, 10), 
#                 strategies_to_use = init_strategies_to_use,
#                 all_strategies = all_strategies,
#                 number_of_conjuncts = random.randint(1, 2)
#             ),
#             sell_dnf = utils.construct_dnf(
#                 trade_type = "sell", 
#                 number_of_disjuncts = random.randint(2, 10), 
#                 strategies_to_use = init_strategies_to_use,
#                 all_strategies = all_strategies,
#                 number_of_conjuncts = random.randint(1, 2)
#             ),
#             strategies_to_use = utils.select_initial_strats(all_strategies, number_of_conjuncts = random.randint(1, 2)),
#             constituent_bot_parameters = constituent_bot_parameters,
#             number_of_disjuncts = random.randint(2, 5),
#             all_strategies = all_strategies,
#             number_of_conjuncts = random.randint(1, 2)
#         ) for i in range(population_size)
#     ]

#     ga_optimiser = EnsembleGeneticAlgorithmOptimizer(
#         ohlcv_df = ohlcv_df_train,
#         trader_agent = trader_agent,
#         trade_signals = trade_signals,
#         fee_percentage = fee_percentage,
#         population_size = population_size,
#         mutation_rate = mutation_rate,
#         num_generations = num_generations,
#         number_of_disjuncts = init_number_of_disjuncts,
#         number_of_conjuncts = init_number_of_conjuncts,
#         all_strategies = all_strategies
#     )

#     best_trader = ga_optimiser.run_genetic_algorithm_ensemble(
#         population = population,
#         n_elite = 2,
#         tournament_size = 10,
#         num_disjuncts_mutate = 1,
#         all_strategies = all_strategies
#     )

#     # best_trade_signals = best_trader.trade_signals()
#     best_trade_signals, _, _ = best_trader.generate_signals()
#     best_number_of_disjuncts = best_trader.number_of_disjuncts
#     best_strategies_to_use = best_trader.strategies_to_use

#     # optimized bot
#     best_final_balance, best_trade_results = utils.execute_trades(
#         trade_signals = best_trade_signals, 
#         fee_percentage = 0.0
#     )
#     utils.plot_trading_simulation(trade_results, "Random Ensemble", color = "blue")
#     utils.plot_trading_simulation(best_trade_results, "Optimized Ensemble", color = "purple")





#     # # instantiate a bot - in this case the stochastic oscillator
#     # ensb_bot = trader_bots.ensemble_bot(
#     #     ohlcv_df = ohlcv_df_train,
#     #     buy_dnf = init_buy_dnf,
#     #     sell_dnf = init_sell_dnf,
#     #     strategies_to_use = init_strategies_to_use,
#     #     constituent_bot_parameters = constituent_bot_parameters,
#     #     number_of_disjuncts = init_number_of_disjuncts,
#     #     all_strategies = all_strategies,
#     #     number_of_conjuncts = init_number_of_conjuncts
#     # )

#     # # generate the trading signals with the bot's technical indicator:
#     # trade_signals, buy_dnf, sell_dnf = ensb_bot.generate_signals()

#     # # instantiate a GeneticAlgorithmOptimizer
#     # ga_optimizer = EnsembleGeneticAlgorithmOptimizer(
#     #     ohlcv_df = ohlcv_df_train,
#     #     trader_agent = ensb_bot,
#     #     trade_signals = trade_signals,
#     #     fee_percentage = fee_percentage,
#     #     population_size = population_size,
#     #     mutation_rate = mutation_rate,
#     #     num_generations = num_generations,
#     #     number_of_disjuncts = init_number_of_disjuncts,
#     #     number_of_conjuncts = init_number_of_conjuncts,
#     #     all_strategies = all_strategies
#     # )

#     # ### Run the Genetic Algorithm ###
#     # best_agent = ga_optimizer.run_genetic_algorithm_ensemble(
#     #     n_elite = 10,
#     #     tournament_size = 50,
#     #     num_disjuncts_mutate = 1,
#     #     all_strategies = all_strategies
#     # )

#     # best_number_of_disjuncts = best_agent.number_of_disjuncts
#     # best_strategies_to_use = best_agent.strategies_to_use

#     # print(f"\n\nbest_number_of_disjuncts:\n{best_number_of_disjuncts}")
#     # print(f"best_strategies_to_use:\n{best_strategies_to_use}\n\n")

#     # # generate the trading signals with the bot's technical indicator:
#     # best_trade_signals, best_buy_dnf, best_sell_dnf = best_agent.generate_signals()

#     # print(f"\nbest buy_dnf:\n{best_buy_dnf}")
#     # print(f"\nbest sell_dnf:\n{best_sell_dnf}\n")

#     # # print(f"Best agent's Trade Signals:\n{best_trade_signals}")

#     # best_final_balance, best_trade_results = utils.execute_trades(best_trade_signals, fee_percentage)

#     # # print(f"Best agent's Trade Results:\n{best_trade_results}")
#     # # print(f"Best agent's Final Balance:\n{best_final_balance}")

#     # utils.plot_trading_simulation(best_trade_results, "Best")
