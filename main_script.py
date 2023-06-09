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

import erwin_ensemble_ga
import utils
import trader_bots

from sklearn.model_selection import train_test_split, KFold
import ga_trader_general as ga


# Base buy-hold strategy
def run_buy_hold_strategy():
    trader_agent = trader_bots.buy_hold_bot(
        ohlcv_df = ohlcv_df_train,
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals,
        fee_percentage = 0.0
    )

    # print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    # utils.plot_trading_simulation(trade_results, "Non Optimized Bollinger Bands", color = "blue")

    return trade_results



# Non-GA Optimised Constituent agents:
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
    # utils.plot_trading_simulation(trade_results, "Non Optimized MACD", color = "red")

    return trade_signals, trader_agent

def run_bollinger_bands_non_optimized():
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

    # print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    # utils.plot_trading_simulation(trade_results, "Non Optimized Bollinger Bands", color = "blue")

    return trade_signals, trader_agent


def run_VWAP_non_optimized():
    trader_agent = trader_bots.VWAP_bot(
        ohlcv_df = ohlcv_df_train,
        window = random.randint(1, 100), 
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    # print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    # utils.plot_trading_simulation(trade_results, "Non Optimized RSI", color = "green")

    return trade_signals, trader_agent


def run_Stoch_Osc_non_optimized():
    trader_agent = trader_bots.stochastic_oscillator_bot(
        ohlcv_df = ohlcv_df_train,
        oscillator_window = random.randint(1, 100), 
        signal_window = random.randint(1, 100),
        overbought_threshold = random.randint(1, 100),
        oversold_threshold = random.randint(1, 100)
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    # print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    # utils.plot_trading_simulation(trade_results, "Non Optimized RSI", color = "green")

    return trade_signals, trader_agent


def run_SAR_non_optimized():
    trader_agent = trader_bots.SAR_bot(
        ohlcv_df = ohlcv_df_train,
        step = random.randint(1, 100), 
        max_step = random.randint(1, 100)
    )

    trade_signals = trader_agent.generate_signals()

    # un-optimized bot
    final_balance, trade_results = utils.execute_trades(
        trade_signals = trade_signals, 
        fee_percentage = 0.0
    )

    # print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    # utils.plot_trading_simulation(trade_results, "Non Optimized RSI", color = "green")

    return trade_signals, trader_agent



def run_RSI_non_optimized():
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

    # print(f"trade_results['portfolio_value'].iloc[-1]: {trade_results['portfolio_value'].iloc[-1]}")
    # utils.plot_trading_simulation(trade_results, "Non Optimized RSI", color = "green")

    return trade_signals, trader_agent


# GA optimised Cnstituent agents:
def run_macd_ga_optimized():

    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized MACD", color = "green")

    print(f"best_trade_results['portfolio_value'].iloc[-1]: {best_trade_results['portfolio_value'].iloc[-1]}")

    return best_trade_results, best_trader


def run_bollinger_bands_ga_optimized():

    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.bollinger_bands_bot(
            ohlcv_df = ohlcv_df_train,
            window = random.randint(1, 100), 
            num_standard_deviations = round(random.uniform(0, 2), 2)
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized Bollinger Bands", color = "green")

    return best_trade_results, best_trader

def run_RSI_ga_optimized():

    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        # trader_agent = trader_agent,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.RSI_bot(
            ohlcv_df = ohlcv_df_train,
            window = random.randint(1, 100), 
            overbought_threshold = random.randint(1, 100),
            oversold_threshold = random.randint(1, 100)
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
    print(f"best window: ")
    # utils.plot_trading_simulation(best_trade_results, "Optimized Bollinger Bands", color = "green")

    return best_trade_results, best_trader


def run_VWAP_ga_optimized():

    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        # trader_agent = trader_agent,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.VWAP_bot(
            ohlcv_df = ohlcv_df_train,
            window = random.randint(1, 100), 
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized VWAP", color = "green")

    return best_trade_results, best_trader

def run_stochastic_oscillator_ga_optimized():
    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        # trader_agent = trader_agent,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.stochastic_oscillator_bot(
            ohlcv_df = ohlcv_df_train,
            oscillator_window = random.randint(1, 100), 
            signal_window = random.randint(1, 100),
            overbought_threshold = random.randint(1, 100),
            oversold_threshold = random.randint(1, 100)
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized VWAP", color = "green")

    return best_trade_results, best_trader

def run_SAR_ga_optimized():
    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.SAR_bot(
            ohlcv_df = ohlcv_df_train,
            step = random.randint(1, 100), 
            max_step = random.randint(1, 100)
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized SAR", color = "green")

    return best_trade_results, best_trader

def run_ROC_ga_optimized():
    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.ROC_bot(
            ohlcv_df = ohlcv_df_train,
            window = random.randint(1, 100),
            buy_threshold = random.randint(1, 100),
            sell_threshold = random.randint(1, 100)
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized ROC", color = "green")

    return best_trade_results, best_trader

def run_Awesome_Oscillator_ga_optimized():
    ga_optimiser = ga.GeneticAlgorithmOptimizer(
        ohlcv_df = ohlcv_df_train,
        fee_percentage = 0.0,
        population_size = population_size,
        mutation_rate = mutation_rate,
        num_generations = num_generations
    )

    population = [
        trader_bots.Awesome_Oscillator_Bot(
            ohlcv_df = ohlcv_df_train,
            window1 = random.randint(1, 100),
            window2 = random.randint(1, 100)
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
    # utils.plot_trading_simulation(best_trade_results, "Optimized ROC", color = "green")

    return best_trade_results, best_trader

def plot_all_optimized_trade_results(
    macd_results,
    bollinger_bands_results,
    rsi_results,
    vwap_results,
    stochastic_results,
    sar_results,
    # obv_following_results,
    # obv_reversal_results,
    roc_results,
    awesome_results,
    buy_hold_results,
    title
    ):
    plt.plot(macd_results.index, macd_results["portfolio_value"], label=f"MACD: {round(macd_results['portfolio_value'].iloc[-1], 2)}", alpha=0.8)
    plt.plot(bollinger_bands_results.index,
             bollinger_bands_results["portfolio_value"], label=f"Bollinger Bands: {round(bollinger_bands_results['portfolio_value'].iloc[-1], 2)}", alpha=0.8)
    plt.plot(rsi_results.index, rsi_results["portfolio_value"],
             label=f"RSI {round(rsi_results['portfolio_value'].iloc[-1], 2)}", alpha=0.8)
    plt.plot(vwap_results.index, vwap_results["portfolio_value"],
             label=f"VWAP {round(vwap_results['portfolio_value'].iloc[-1], 2)}", alpha=0.8)
    plt.plot(stochastic_results.index,
             stochastic_results["portfolio_value"], label=f"Stochastic {round(stochastic_results['portfolio_value'].iloc[-1],2)}", alpha=0.8)
    plt.plot(sar_results.index, sar_results["portfolio_value"],
             label=f"SAR {round(sar_results['portfolio_value'].iloc[-1],2)}", alpha=0.8)
    plt.plot(roc_results.index, roc_results["portfolio_value"],
             label=f"ROC {round(roc_results['portfolio_value'].iloc[-1],2)}", alpha=0.8)
    plt.plot(awesome_results.index, awesome_results["portfolio_value"], label=f"Awesome Oscillator {round(awesome_results['portfolio_value'].iloc[-1],2)}", alpha=0.8)
    plt.plot(buy_hold_results.index, buy_hold_results["portfolio_value"], label=f"Buy-Hold Strategy: {round(buy_hold_results['portfolio_value'].iloc[-1],2)}", alpha=0.8)


    plt.xlabel('Day')
    plt.ylabel('Portfolio')
    plt.title(f'{title}')
    plt.legend()
    plt.show()


# Ensemble Agent Run on Non-GA Optimised Constituent Agents:
def run_ensemble_bots_non_optimal_and_optimal(Non_Optimized_constituent_bot_parameters, Optimized_constituent_bot_parameters):
    best_ensemble_bot_non_optimised = erwin_ensemble_ga.ensemble_ga(constituent_bot_parameters=Non_Optimized_constituent_bot_parameters, population_size=100, number_of_generations=20, mutation_rate=0.5)
    best_ensemble_bot_optimised = erwin_ensemble_ga.ensemble_ga(constituent_bot_parameters=Optimized_constituent_bot_parameters, population_size=100, number_of_generations=20, mutation_rate=0.5)
    return best_ensemble_bot_non_optimised, best_ensemble_bot_optimised

if __name__ == "__main__":

    # ohlcv_df = utils.get_daily_ohlcv_data()
    ohlcv_df_train = utils.get_daily_ohlcv_data()
    # ohlcv_df_train, ohlcv_df_test = train_test_split(ohlcv_df, test_size = 0.2, shuffle = False)

    fee_percentage = 0.02

    population_size = 5
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
    # run_bollinger_bands_non_optimized()

    ### GA OPTIMISED CONSTITUENT BOTS ###
    # run_macd_ga_optimized()
    # run_bollinger_bands_ga_optimized()
    # run_RSI_ga_optimized()
    # run_VWAP_ga_optimized()
    # run_stochastic_oscillator_ga_optimized()
    # run_SAR_ga_optimized()
    # run_ROC_ga_optimized()
    # run_Awesome_Oscillator_ga_optimized()

    macd_results, best_MACD_trader = run_macd_ga_optimized()
    bollinger_bands_results, best_Bol_Band_trader = run_bollinger_bands_ga_optimized()
    rsi_results, best_RSI_trader = run_RSI_ga_optimized()
    vwap_results, best_VWAP_trader = run_VWAP_ga_optimized()
    stochastic_results, best_Stoch_Osc_trader = run_stochastic_oscillator_ga_optimized()
    sar_results, best_SAR_trader = run_SAR_ga_optimized()
    # obv_following_results = obv_following_results()
    # obv_reversal_results = obv_reversal_results()
    roc_results, best_ROC_trader = run_ROC_ga_optimized()
    awesome_results, best_Awe_Osc_trader = run_Awesome_Oscillator_ga_optimized()

    # ensemble_results = run_ensemble_non_optimal_constituents()
    buy_hold_results = run_buy_hold_strategy()

    plot_all_optimized_trade_results(
        macd_results,
        bollinger_bands_results,
        rsi_results,
        vwap_results,
        stochastic_results,
        sar_results,
        # obv_following_results,
        # obv_reversal_results,
        roc_results,
        awesome_results,
        buy_hold_results,
        "Trade Results for Optimized Constituent Bots"
    )

    ######## Best MACD params:
    best_macd_slow_window = best_MACD_trader.slow_window
    best_macd_fast_window = best_MACD_trader.fast_window
    best_macd_signal_window = best_MACD_trader.signal_window

    # print(f"best_macd_slow_window: {best_macd_slow_window}")
    # print(f"best_macd_fast_window: {best_macd_fast_window}")
    # print(f"best_macd_signal_window: {best_macd_signal_window}")

    ######## Best Bollinger Bands Params:
    best_bollinger_window = best_Bol_Band_trader.window
    best_bollinger_num_standard_deviations = best_Bol_Band_trader.num_standard_deviations

    # print(f"best_bollinger_window: {best_bollinger_window}")
    # print(f"best_bollinger_num_standard_deviations: {best_bollinger_num_standard_deviations}")

    ######## Best RSI params:
    best_rsi_window = best_RSI_trader.window
    best_rsi_overbought_threshold = best_RSI_trader.overbought_threshold
    best_rsi_oversold_threshold = best_RSI_trader.oversold_threshold

    # print(f"best_rsi_window: {best_rsi_window}")
    # print(f"best_rsi_overbought_threshold: {best_rsi_overbought_threshold}")
    # print(f"best_rsi_oversold_threshold: {best_rsi_oversold_threshold}")

    ######## Best VWAP Params:
    best_vwap_window = best_VWAP_trader.window

    # print(f"best_vwap_window: {best_vwap_window}")

    ######## Best Stochastic Oscillator Params:
    # oscillator_window, signal_window, overbought_threshold, oversold_threshold
    best_stoch_osc_oscillator_window = best_Stoch_Osc_trader.oscillator_window
    best_stoch_osc_signal_window = best_Stoch_Osc_trader.signal_window
    best_stoch_osc_overbought_threshold = best_Stoch_Osc_trader.overbought_threshold
    best_stoch_osc_oversold_threshold = best_Stoch_Osc_trader.oversold_threshold

    # print(f"best_stoch_osc_oscillator_window: {best_stoch_osc_oscillator_window}")
    # print(f"best_stoch_osc_signal_window: {best_stoch_osc_signal_window}")
    # print(f"best_stoch_osc_overbought_threshold: {best_stoch_osc_overbought_threshold}")
    # print(f"best_stoch_osc_oversold_threshold: {best_stoch_osc_oversold_threshold}")

    ######## Best SAR Params:
    best_sar_step = best_SAR_trader.step
    best_sar_max_step = best_SAR_trader.max_step

    # print(f"best_sar_step: {best_sar_step}")
    # print(f"best_sar_max_step: {best_sar_max_step}")

    ######## Best ROC Params:
    best_roc_window = best_ROC_trader.window
    best_roc_buy_threshold = best_ROC_trader.buy_threshold
    best_roc_sell_threshold = best_ROC_trader.sell_threshold

    # print(f"best_roc_window: {best_roc_window}")
    # print(f"best_roc_buy_threshold: {best_roc_buy_threshold}")
    # print(f"best_roc_sell_threshold: {best_roc_sell_threshold}")


    ######## Best Awesome Oscillator Params:
    best_Awe_Osc_window1 = best_Awe_Osc_trader.window1
    best_Awe_Osc_window2 = best_Awe_Osc_trader.window2

    # print(f"best_Awe_Osc_window1: {best_Awe_Osc_window1}")
    # print(f"best_Awe_Osc_window2: {best_Awe_Osc_window2}")



    macd_results, best_MACD_trader = run_macd_ga_optimized()
    bollinger_bands_results, best_Bol_Band_trader = run_bollinger_bands_ga_optimized()
    rsi_results, best_RSI_trader = run_RSI_ga_optimized()
    vwap_results, best_VWAP_trader = run_VWAP_ga_optimized()
    stochastic_results, best_Stoch_Osc_trader = run_stochastic_oscillator_ga_optimized()
    sar_results, best_SAR_trader = run_SAR_ga_optimized()
    # obv_following_results = obv_following_results()
    # obv_reversal_results = obv_reversal_results()
    roc_results, best_ROC_trader = run_ROC_ga_optimized()
    awesome_results, best_Awe_Osc_trader = run_Awesome_Oscillator_ga_optimized()

    # ensemble_results = run_ensemble_non_optimal_constituents()
    buy_hold_results = run_buy_hold_strategy()

    plot_all_optimized_trade_results(
        macd_results,
        bollinger_bands_results,
        rsi_results,
        vwap_results,
        stochastic_results,
        sar_results,
        # obv_following_results,
        # obv_reversal_results,
        roc_results,
        awesome_results,
        buy_hold_results,
        "Trade Results for Optimized Constituent Bots"
    )


    Optimized_MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': best_macd_slow_window, 'fast_window': best_macd_fast_window, 'signal_window': best_macd_signal_window}
    Optimized_Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': best_bollinger_window, 'num_standard_deviations': best_bollinger_num_standard_deviations}
    Optimized_RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': best_rsi_overbought_threshold, 'oversold_threshold': best_rsi_oversold_threshold, 'window': best_rsi_window}
    Optimized_VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': best_vwap_window}
    Optimized_Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': best_stoch_osc_oscillator_window, 'signal_window': best_stoch_osc_signal_window, 'overbought_threshold': best_stoch_osc_overbought_threshold, 'oversold_threshold': best_stoch_osc_oversold_threshold}
    Optimized_SAR_parameters = {'bot_name': 'SAR_bot', 'step': best_sar_step, 'max_step': best_sar_max_step}
    Optimized_OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
    Optimized_OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
    Optimized_ROC_parameters = {'bot_name': 'ROC_bot', 'window': best_roc_window, 'buy_threshold': best_roc_buy_threshold, 'sell_threshold': best_roc_sell_threshold}
    Optimized_Awesome_Osillator = {'bot_name': 'Awesome_Oscillator_Bot', 'window1': best_Awe_Osc_window1 , 'window2': best_Awe_Osc_window2}

    Optimized_constituent_bot_parameters = [ 
        Optimized_Bollinger_Bands_parameters, 
        Optimized_MACD_parameters,
        Optimized_RSI_parameters, 
        Optimized_VWAP_parameters, 
        Optimized_Stochastic_Oscillator_parameters,
        Optimized_OBV_trend_following_parameters,
        Optimized_SAR_parameters,
        Optimized_OBV_trend_reversal_parameters,
        Optimized_ROC_parameters,
        Optimized_Awesome_Osillator
    ]

    # run = 1
    # with open("Optimized_constituent_bot_parameters_{run}.txt", "w") as f:
    #     for item in Optimized_constituent_bot_parameters:
    #         f.write(str(item) + "\n")
    # f.close()

    Non_Optimized_MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
    Non_Optimized_Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
    Non_Optimized_RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
    Non_Optimized_VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
    Non_Optimized_Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14, 'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}
    Non_Optimized_SAR_parameters = {'bot_name': 'SAR_bot', 'step': 0.02, 'max_step': 0.2}
    Non_Optimized_OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
    Non_Optimized_OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
    Non_Optimized_ROC_parameters = {'bot_name': 'ROC_bot', 'window': 12, 'buy_threshold': 5, 'sell_threshold': -5}
    Non_Optimized_Awesome_Osillator = {'bot_name': 'Awesome_Oscillator_Bot', 'window1': 5 , 'window2': 34}

    Non_Optimized_constituent_bot_parameters = [ 
        Non_Optimized_Bollinger_Bands_parameters, 
        Non_Optimized_MACD_parameters,
        Non_Optimized_RSI_parameters, 
        Non_Optimized_VWAP_parameters, 
        Non_Optimized_Stochastic_Oscillator_parameters,
        Non_Optimized_OBV_trend_following_parameters,
        Non_Optimized_SAR_parameters,
        Non_Optimized_OBV_trend_reversal_parameters,
        Non_Optimized_ROC_parameters,
        Non_Optimized_Awesome_Osillator
    ]

    ### ENSEMBLE BOTS ###
    run_ensemble_bots_non_optimal_and_optimal(Non_Optimized_constituent_bot_parameters, Optimized_constituent_bot_parameters)