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
import ga_trader_general as ga



def run_ensemble_non_optimal_constituents():

    MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
    Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
    RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
    VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
    Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14, 'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}
    SAR_parameters = {'bot_name': 'SAR_bot', 'step': 0.02, 'max_step': 0.2}
    OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
    OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
    ROC_parameters = {'bot_name': 'ROC_bot', 'window': 12, 'buy_threshold': 5, 'sell_threshold': -5}
    Awesome_Osillator = {'bot_name': 'Awesome_Oscillator_Bot', 'window1': 5 , 'window2': 34}

    constituent_bot_parameters = [ 
        Bollinger_Bands_parameters, 
        MACD_parameters,
        RSI_parameters, 
        VWAP_parameters, 
        Stochastic_Oscillator_parameters,
        OBV_trend_following_parameters,
        SAR_parameters,
        OBV_trend_reversal_parameters,
        ROC_parameters,
        Awesome_Osillator
    ]

    all_strategies = [
        'MACD_bot', 'bollinger_bands_bot', 
        'RSI_bot', 'VWAP_bot', 
        'stochastic_oscillator_bot', 
        'SAR_bot',
        'OBV_trend_following_bot',
        'OBV_trend_reversal_bot',
        'ROC_bot', 'Awesome_Oscillator_Bot'
    ]

    # init strats randomly 
    init_strategies_to_use = utils.select_initial_strats(all_strategies, init_number_of_conjuncts)

    # construct initial buy_dnf
    init_buy_dnf = utils.construct_dnf(
        trade_type = "buy", 
        number_of_disjuncts = init_number_of_disjuncts, 
        strategies_to_use = init_strategies_to_use,
        all_strategies = all_strategies,
        number_of_conjuncts = init_number_of_conjuncts
    )

    # construct initial sell_dnf
    init_sell_dnf = utils.construct_dnf(
        trade_type = "sell", 
        number_of_disjuncts = init_number_of_disjuncts, 
        strategies_to_use = init_strategies_to_use,
        all_strategies = all_strategies,
        number_of_conjuncts = init_number_of_conjuncts
    )

    trader_agent = trader_bots.ensemble_bot(
        ohlcv_df = ohlcv_df_train,
        buy_dnf = init_buy_dnf,
        sell_dnf = init_sell_dnf,
        strategies_to_use = init_strategies_to_use,
        constituent_bot_parameters = constituent_bot_parameters,
        number_of_disjuncts = init_number_of_disjuncts,
        all_strategies = all_strategies,
        number_of_conjuncts = init_number_of_conjuncts
    )

    trade_signals, _, _ = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    utils.plot_trading_simulation(trade_results, "Random Ensemble", color = "blue")

    population = [
        trader_bots.ensemble_bot(
            ohlcv_df = ohlcv_df_train,
            buy_dnf = utils.construct_dnf(
                trade_type = "buy", 
                number_of_disjuncts = random.randint(2, 10), 
                strategies_to_use = init_strategies_to_use,
                all_strategies = all_strategies,
                number_of_conjuncts = random.randint(1, 2)
            ),
            sell_dnf = utils.construct_dnf(
                trade_type = "sell", 
                number_of_disjuncts = random.randint(2, 10), 
                strategies_to_use = init_strategies_to_use,
                all_strategies = all_strategies,
                number_of_conjuncts = random.randint(1, 2)
            ),
            strategies_to_use = utils.select_initial_strats(all_strategies, number_of_conjuncts = random.randint(1, 2)),
            constituent_bot_parameters = constituent_bot_parameters,
            number_of_disjuncts = random.randint(2, 5),
            all_strategies = all_strategies,
            number_of_conjuncts = random.randint(1, 2)
        ) for i in range(population_size)
    ]

    ga_optimiser = ga.EnsembleGeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        trader_agent = trader_agent,
        trade_signals = trade_signals,
        fee_percentage = fee_percentage,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations,
        number_of_disjuncts = init_number_of_disjuncts,
        number_of_conjuncts = init_number_of_conjuncts,
        all_strategies = all_strategies
    )

    best_trader = ga_optimiser.brute_force_search_ensemble(population)

    best_trade_signals, _, _ = best_trader.generate_signals()

    # un-optimized bot
    best_final_balance, best_trade_results = utils.execute_trades(
        trade_signals = best_trade_signals, 
        fee_percentage = 0.0
    )

    utils.plot_trading_simulation(best_trade_results, "Random Ensemble", color = "orange")

def run_macd_non_optimized():
    trader_agent = trader_bots.MACD_bot(
        ohlcv_df = ohlcv_df_train,
        slow_window = slow_window, 
        fast_window = fast_window, 
        signal_window = signal_window
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    utils.plot_trading_simulation(trade_results, "Non Optimized MACD", color = "red")

def run_bolliger_band_non_optimized():
    trader_agent = trader_bots.bollinger_bands_bot(
        ohlcv_df = ohlcv_df_train,
        window = window, 
        num_standard_deviations = num_standard_deviations
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    utils.plot_trading_simulation(trade_results, "Non Optimized Bollinger Bands", color = "blue")

def run_bolliger_band_non_optimized():
    trader_agent = trader_bots.RSI_bot(
        ohlcv_df = ohlcv_df_train,
        window = window, 
        overbought_threshold = overbought_threshold, 
        oversold_threshold = oversold_threshold,
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    utils.plot_trading_simulation(trade_results, "Non Optimized Bollinger Bands", color = "green")

def run_macd_ga_optimized():
    trader_agent = trader_bots.MACD_bot(
        ohlcv_df = ohlcv_df_train,
        slow_window = slow_window, 
        fast_window = fast_window, 
        signal_window = signal_window
    )

    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        trader_agent = trader_agent,
        trade_signals = trade_signals,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.MACD_bot(
            ohlcv_df = ohlcv_df_train,
            slow_window = random.randint(1, 100),
            fast_window = random.randint(1, 100),
            signal_window = random.randint(1, 100),
        ) for _ in range(population_size)
    ] 

    best_trader = ga_optimiser.run_genetic_algorithm(
        population = population,
        n_elite = 2, 
        tournament_size = 5
    )

    # best_trade_signals = best_trader.trade_signals()
    best_trade_signals = best_trader.generate_signals()

    # optimized bot
    best_final_balance, best_trade_results = utils.execute_trades(
        trade_signals = best_trade_signals, 
        fee_percentage = 0.0
    )
    utils.plot_trading_simulation(trade_results, "Non Optimized MACD", color = "red")
    utils.plot_trading_simulation(best_trade_results, "Optimized MACD", color = "green")


if __name__ == "__main__":

    # ohlcv_df = utils.get_daily_ohlcv_data()
    ohlcv_df_train = utils.get_daily_ohlcv_data()
    # ohlcv_df_train, ohlcv_df_test = train_test_split(ohlcv_df, test_size = 0.2, shuffle = False)

    fee_percentage = 0.02

    population_size = 20
    mutation_rate = 0.01
    num_generations = 2
    n_elite = 2
    init_number_of_disjuncts = 5
    init_number_of_conjuncts = 2

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
    slow_window = 26
    fast_window = 12
    signal_window = 9

    ### CONSTITUENT BOTS ###

    # run_macd_non_optimized()
    run_bolliger_band_non_optimized()


    ### ENSEMBLE BOTS ###
    # run_ensemble_non_optimal_constituents()

   




########################################################

    # population = [
    #     trader_bots.ensemble_bot(
    #         ohlcv_df = ohlcv_df_train,
    #         buy_dnf = utils.construct_dnf(
    #             trade_type = "buy", 
    #             number_of_disjuncts = random.randint(2, 10), 
    #             strategies_to_use = init_strategies_to_use,
    #             all_strategies = all_strategies,
    #             number_of_conjuncts = random.randint(1, 2)
    #         ),
    #         sell_dnf = utils.construct_dnf(
    #             trade_type = "sell", 
    #             number_of_disjuncts = random.randint(2, 10), 
    #             strategies_to_use = init_strategies_to_use,
    #             all_strategies = all_strategies,
    #             number_of_conjuncts = random.randint(1, 2)
    #         ),
    #         strategies_to_use = utils.select_initial_strats(all_strategies, number_of_conjuncts = random.randint(1, 2)),
    #         constituent_bot_parameters = constituent_bot_parameters,
    #         number_of_disjuncts = random.randint(2, 5),
    #         all_strategies = all_strategies,
    #         number_of_conjuncts = random.randint(1, 2)
    #     ) for i in range(population_size)
    # ]

    # ga_optimiser = ga.EnsembleGeneticAlgorithmOptimizer(
    #     ohlcv_df = ohlcv_df_train,
    #     trader_agent = trader_agent,
    #     trade_signals = trade_signals,
    #     fee_percentage = fee_percentage,
    #     population_size = population_size,
    #     mutation_rate = mutation_rate,
    #     num_generations = num_generations,
    #     number_of_disjuncts = init_number_of_disjuncts,
    #     number_of_conjuncts = init_number_of_conjuncts,
    #     all_strategies = all_strategies
    # )

    # best_trader = ga_optimiser.run_genetic_algorithm_ensemble(
    #     population = population,
    #     n_elite = 2,
    #     tournament_size = 10,
    #     num_disjuncts_mutate = 1,
    #     all_strategies = all_strategies
    # )

    # # best_trade_signals = best_trader.trade_signals()
    # best_trade_signals, _, _ = best_trader.generate_signals()
    # best_number_of_disjuncts = best_trader.number_of_disjuncts
    # best_strategies_to_use = best_trader.strategies_to_use

    # # optimized bot
    # best_final_balance, best_trade_results = utils.execute_trades(
    #     trade_signals = best_trade_signals, 
    #     fee_percentage = 0.0
    # )
    # utils.plot_trading_simulation(trade_results, "Random Ensemble", color = "blue")
    # utils.plot_trading_simulation(best_trade_results, "Optimized Ensemble", color = "purple")





    # # instantiate a bot - in this case the stochastic oscillator
    # ensb_bot = trader_bots.ensemble_bot(
    #     ohlcv_df = ohlcv_df_train,
    #     buy_dnf = init_buy_dnf,
    #     sell_dnf = init_sell_dnf,
    #     strategies_to_use = init_strategies_to_use,
    #     constituent_bot_parameters = constituent_bot_parameters,
    #     number_of_disjuncts = init_number_of_disjuncts,
    #     all_strategies = all_strategies,
    #     number_of_conjuncts = init_number_of_conjuncts
    # )

    # # generate the trading signals with the bot's technical indicator:
    # trade_signals, buy_dnf, sell_dnf = ensb_bot.generate_signals()

    # # instantiate a GeneticAlgorithmOptimizer
    # ga_optimizer = ga.EnsembleGeneticAlgorithmOptimizer(
    #     ohlcv_df = ohlcv_df_train,
    #     trader_agent = ensb_bot,
    #     trade_signals = trade_signals,
    #     fee_percentage = fee_percentage,
    #     population_size = population_size,
    #     mutation_rate = mutation_rate,
    #     num_generations = num_generations,
    #     number_of_disjuncts = init_number_of_disjuncts,
    #     number_of_conjuncts = init_number_of_conjuncts,
    #     all_strategies = all_strategies
    # )

    # ### Run the Genetic Algorithm ###
    # best_agent = ga_optimizer.run_genetic_algorithm_ensemble(
    #     n_elite = 10,
    #     tournament_size = 50,
    #     num_disjuncts_mutate = 1,
    #     all_strategies = all_strategies
    # )

    # best_number_of_disjuncts = best_agent.number_of_disjuncts
    # best_strategies_to_use = best_agent.strategies_to_use

    # print(f"\n\nbest_number_of_disjuncts:\n{best_number_of_disjuncts}")
    # print(f"best_strategies_to_use:\n{best_strategies_to_use}\n\n")

    # # generate the trading signals with the bot's technical indicator:
    # best_trade_signals, best_buy_dnf, best_sell_dnf = best_agent.generate_signals()

    # print(f"\nbest buy_dnf:\n{best_buy_dnf}")
    # print(f"\nbest sell_dnf:\n{best_sell_dnf}\n")

    # # print(f"Best agent's Trade Signals:\n{best_trade_signals}")

    # best_final_balance, best_trade_results = utils.execute_trades(best_trade_signals, fee_percentage)

    # # print(f"Best agent's Trade Results:\n{best_trade_results}")
    # # print(f"Best agent's Final Balance:\n{best_final_balance}")

    # utils.plot_trading_simulation(best_trade_results, "Best")
