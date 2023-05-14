### Imports

import utils
import trader_bots
from sklearn.model_selection import train_test_split
import random

# DEFINE PARAMETERS (FOR EVERYTHING)
###############################################################################

### GA parameters
def ensemble_ga(constituent_bot_parameters, population_size=100, number_of_generations=20, mutation_rate=0.5):
    injected_population_size = 20

    ### Constituent bot parameters

    MACD_parameters = {'bot_name': 'MACD_bot', 'slow_window': 26, 'fast_window': 12, 'signal_window': 9}
    Bollinger_Bands_parameters = {'bot_name': 'bollinger_bands_bot', 'window': 20, 'num_standard_deviations': 2.5}
    RSI_parameters = {'bot_name': 'RSI_bot', 'overbought_threshold': 70, 'oversold_threshold': 30, 'window': 14}
    VWAP_parameters = {'bot_name': 'VWAP_bot', 'window': 20}
    Stochastic_Oscillator_parameters = {'bot_name': 'stochastic_oscillator_bot', 'oscillator_window': 14,
                                        'signal_window': 3, 'overbought_threshold': 80, 'oversold_threshold': 20}
    SAR_parameters = {'bot_name': 'SAR_bot', 'step': 0.02, 'max_step': 0.2}
    OBV_trend_following_parameters = {'bot_name': 'OBV_trend_following_bot'}
    OBV_trend_reversal_parameters = {'bot_name': 'OBV_trend_reversal_bot'}
    ROC_parameters = {'bot_name': 'ROC_bot', 'window': 12, 'buy_threshold': 5, 'sell_threshold': -5}

    ### Ensemble bot parameters


    all_parameters = [
        MACD_parameters,
        Bollinger_Bands_parameters,
        RSI_parameters,
        VWAP_parameters,
        Stochastic_Oscillator_parameters,
        OBV_trend_following_parameters,
        SAR_parameters,
        OBV_trend_reversal_parameters,
        ROC_parameters
    ]
    min_terms_in_conjunction = 1
    max_terms_in_conjunction = 4
    min_conjunctions_in_dnf = 1
    max_conjunctions_in_dnf = 4

    ### Simulation parameters

    fee_percentage = 0.02
    test_size = 0.01

    # GENERATION 0 (INITIAL POPULATION)
    ###############################################################################

    ### Get data

    ohlcv_df = utils.get_daily_ohlcv_data()
    ohlcv_df_train, ohlcv_df_test = train_test_split(ohlcv_df, test_size=test_size, shuffle=False)

    # Check:
    # print("Training data: ")
    # print(ohlcv_df_train.to_string())
    # print("\nTest data:")
    # print(ohlcv_df_test.to_string())


    ### Determine constituent bot signals for training data

    # all_bot_signals, all_bot_names = utils.initialise_bots(ohlcv_df_train, all_parameters)
    all_bot_signals, all_bot_names = utils.initialise_bots(ohlcv_df_train, constituent_bot_parameters)

    # Check:
    # print(all_bot_signals)
    # print(all_bot_names)


    ### Initialise ensemble bot population with training data

    population = [trader_bots.ensemble_bot(
        ohlcv_df=ohlcv_df_train,
        all_bot_signals=all_bot_signals,
        all_bot_names=all_bot_names,
        min_terms_in_conjunction=min_terms_in_conjunction,
        max_terms_in_conjunction=max_terms_in_conjunction,
        min_conjunctions_in_dnf=min_conjunctions_in_dnf,
        max_conjunctions_in_dnf=max_conjunctions_in_dnf
    ) for ensemble_bot in range(0, population_size)]

    population_with_info = []

    for ensemble_bot in population:
        buy_dnf, sell_dnf, trade_signals = ensemble_bot.generate_signals()
        final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
        ensemble_bot_with_info = [ensemble_bot, buy_dnf, sell_dnf, trade_results, final_balance]
        population_with_info.append(ensemble_bot_with_info)

    # Check:
    # print(population_with_info)


    ### Rank population_with_info by final_balance from highest to lowest

    population_with_info = sorted(population_with_info, key=lambda x: x[4], reverse=True)

    ### Print message

    print("\nInitial population (Generation 0):\n")
    print(f"Best ensemble bot's buy_dnf: \n{population_with_info[0][1]}\n")
    print(f"Best ensemble bot's sell_dnf: \n{population_with_info[0][2]}\n")
    print(f"Best ensemble bot's final_balance: {population_with_info[0][4]}\n")

    # Check:
    # print(population_with_info_sorted)


    # GENERATIONS >= 1 (THE EVOLUTION)
    ###############################################################################

    for generation in range(1, number_of_generations + 1):

        ### Perform crossover to generate offspring

        offspring_with_info = []

        for crossover in range(0, len(population_with_info)):
            # Select two parents from the population randomly.
            parent1_idx = random.randint(0, len(population_with_info) - 1)
            parent2_idx = random.randint(0, len(population_with_info) - 1)
            while parent2_idx == parent1_idx:
                parent2_idx = random.randint(0, len(population_with_info) - 1)

            parent1 = population_with_info[parent1_idx]
            parent2 = population_with_info[parent2_idx]

            # print(f"parent1: \n{parent1[1]} \n{parent1[2]}\n")
            # print(f"parent2: \n{parent2[1]} \n{parent2[2]}\n")

            # Let child inherit a buy_dnf from its parents.
            if random.random() < 0.5:
                child_buy_dnf = parent1[1]
            else:
                child_buy_dnf = parent2[1]

            # Let child inherit a sell_dnf from its parents.
            if random.random() < 0.5:
                child_sell_dnf = parent1[2]
            else:
                child_sell_dnf = parent2[2]

            child_ensemble_bot = trader_bots.ensemble_bot(
                ohlcv_df=ohlcv_df_train,
                all_bot_signals=all_bot_signals,
                all_bot_names=all_bot_names,
                min_terms_in_conjunction=min_terms_in_conjunction,
                max_terms_in_conjunction=max_terms_in_conjunction,
                min_conjunctions_in_dnf=min_conjunctions_in_dnf,
                max_conjunctions_in_dnf=max_conjunctions_in_dnf
            )

            child_buy_dnf, child_sell_dnf, child_trade_signals = child_ensemble_bot.generate_signals(child_buy_dnf,
                                                                                                     child_sell_dnf)
            child_final_balance, child_trade_results = utils.execute_trades(child_trade_signals, fee_percentage)
            child_ensemble_bot_with_info = [child_ensemble_bot, child_buy_dnf, child_sell_dnf, child_trade_results,
                                            child_final_balance]
            # print(f"child: \n{child_ensemble_bot_with_info[1]} \n{child_ensemble_bot_with_info[2]}\n")

            offspring_with_info.append(child_ensemble_bot_with_info)

        # Check: embedded.

        ### Perform random mutation of a conjunction for offspring

        for index in range(0, len(offspring_with_info) - 1):
            ensemble_bot = offspring_with_info[index]

            # print(f"original: \n{ensemble_bot[1]} \n{ensemble_bot[2]}\n")
            if random.random() < mutation_rate:
                # Get the ensemble bot's current dnfs
                buy_dnf = ensemble_bot[1]
                sell_dnf = ensemble_bot[2]

                # 50/50 chance of mutating a conjunction in the ensemble bot's buy_dnf or sell_dnf
                if random.random() < 0.5:
                    buy_dnf = utils.mutate_dnf(buy_dnf, all_bot_names)
                else:
                    sell_dnf = utils.mutate_dnf(sell_dnf, all_bot_names)

                # Instantiate the mutated version of the offspring bot
                mutated_ensemble_bot = trader_bots.ensemble_bot(
                    ohlcv_df=ohlcv_df_train,
                    all_bot_signals=all_bot_signals,
                    all_bot_names=all_bot_names,
                    min_terms_in_conjunction=min_terms_in_conjunction,
                    max_terms_in_conjunction=max_terms_in_conjunction,
                    min_conjunctions_in_dnf=min_conjunctions_in_dnf,
                    max_conjunctions_in_dnf=max_conjunctions_in_dnf)

                buy_dnf, sell_dnf, trade_signals = mutated_ensemble_bot.generate_signals(buy_dnf, sell_dnf)
                final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
                mutated_ensemble_bot_with_info = [mutated_ensemble_bot, buy_dnf, sell_dnf, trade_results, final_balance]
                # print(f"mutated: \n{mutated_ensemble_bot_with_info[1]} \n{mutated_ensemble_bot_with_info[2]}\n")

                # Replace the non-mutated offspring bot with the mutated one.
                offspring_with_info.pop(index)
                offspring_with_info.append(mutated_ensemble_bot_with_info)

        # Check: embedded.

        ### Inject random immigrants

        injected_population = [trader_bots.ensemble_bot(
            ohlcv_df=ohlcv_df_train,
            all_bot_signals=all_bot_signals,
            all_bot_names=all_bot_names,
            min_terms_in_conjunction=min_terms_in_conjunction,
            max_terms_in_conjunction=max_terms_in_conjunction,
            min_conjunctions_in_dnf=min_conjunctions_in_dnf,
            max_conjunctions_in_dnf=max_conjunctions_in_dnf)
            for ensemble_bot in range(0, injected_population_size)]

        injected_population_with_info = []

        for ensemble_bot in injected_population:
            buy_dnf, sell_dnf, trade_signals = ensemble_bot.generate_signals()
            final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
            ensemble_bot_with_info = [ensemble_bot, buy_dnf, sell_dnf, trade_results, final_balance]
            injected_population_with_info.append(ensemble_bot_with_info)

        ### Merge original, offspring, and random injected populations

        merged_population_with_info = population_with_info + offspring_with_info + injected_population_with_info

        ### Rank the merged populaation and reduce it to the specified population size

        merged_population_with_info_sorted = sorted(merged_population_with_info, key=lambda x: x[4], reverse=True)
        population_with_info = merged_population_with_info_sorted[0:population_size]

        ### Print message

        print(f"\nGeneration {generation}:\n")
        print(f"Best ensemble bot's buy_dnf: \n{population_with_info[0][1]}\n")
        print(f"Best ensemble bot's sell_dnf: \n{population_with_info[0][2]}\n")
        print(f"Best ensemble bot's final_balance: {population_with_info[0][4]}\n")

    # ASSESSMENT OF FINAL RESULT (CURRENTLY JUST PLOTTING)
    ###############################################################################

    ### Plot the best ensemble bot from the last generation

    utils.plot_trading_simulation([population_with_info[0][3]], ["Ensemble Bot"],
                                  f"Best Ensemble Bot After {number_of_generations} Generations (Population = {population_size})")

    return population_with_info[0][3]
