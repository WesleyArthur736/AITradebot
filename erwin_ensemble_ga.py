import utils
import trader_bots
from sklearn.model_selection import train_test_split
import random


def crossover(ohlcv_df, fee_percentage, all_bot_signals, all_bot_names, population_with_info, tournament_selection_pool_size = 10):

    min_terms_in_conjunction = 1
    max_terms_in_conjunction = 4
    min_conjunctions_in_dnf = 1
    max_conjunctions_in_dnf = 4

    offspring_with_info = []

    for _ in range(0, len(population_with_info)):

        # create a pool of 'tournament_selection_pool_size' parents 
        parents_pool = [
            parent_candidate for parent_candidate in sorted(population_with_info[:tournament_selection_pool_size], key=lambda x: x[4], reverse=True)
        ]
        # sort this 'parents_pool' by the fitness of the parents, fittest first-to-last 
        parents_pool = sorted(parents_pool, key=lambda x: x[4], reverse=True)

        # the selected parents are simply the first to parents of the sorted 'parents_pool' 
        parent1 = parents_pool[0]
        parent2 = parents_pool[1]

        # Let child inherit a buy_dnf from its parents.
        if random.random() < 0.5:
            child_buy_dnf = parent1[1]
            child_sell_dnf = parent2[2]
        else:
            child_buy_dnf = parent1[2]
            child_sell_dnf = parent2[1]

        # instantiate a child bot 
        child_ensemble_bot = trader_bots.ensemble_bot(
            ohlcv_df = ohlcv_df,
            all_bot_signals = all_bot_signals,
            all_bot_names = all_bot_names,
            min_terms_in_conjunction = min_terms_in_conjunction,
            max_terms_in_conjunction = max_terms_in_conjunction,
            min_conjunctions_in_dnf = min_conjunctions_in_dnf,
            max_conjunctions_in_dnf = max_conjunctions_in_dnf
        )

        _, _, child_trade_signals = child_ensemble_bot.generate_signals(child_buy_dnf, child_sell_dnf)
        child_final_balance, child_trade_results = utils.execute_trades(child_trade_signals, fee_percentage)
        child_ensemble_bot_with_info = [child_ensemble_bot, child_buy_dnf, child_sell_dnf, child_trade_results, child_final_balance]

        offspring_with_info.append(child_ensemble_bot_with_info)

    return offspring_with_info


def ensemble_ga(constituent_bot_parameters, population_size = 100, number_of_generations = 20, mutation_rate = 0.5):

    # the number of "random immigrants" to inject in each generation
    injected_population_size = 20

    # the minimum number of trade signals to include in each conjunction
    min_terms_in_conjunction = 1

    # the maximum number of trade signals to include in each conjunction
    max_terms_in_conjunction = 4

    # the minimum number of conjunctions to include in each conjunction
    min_conjunctions_in_dnf = 1

    # the maximum number of conjunctions to include in each conjunction
    max_conjunctions_in_dnf = 4

    # Simulation parameters
    fee_percentage = 0.02
    test_size = 0.01

    # Get the ohclv data with the reevant helper function
    ohlcv_df = utils.get_daily_ohlcv_data()
    # ohlcv_df, ohlcv_df_test = train_test_split(ohlcv_df, test_size=test_size, shuffle=False) # don't do this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Determine constituent bot signals for training data
    all_bot_signals, all_bot_names = utils.initialise_bots(ohlcv_df, constituent_bot_parameters)

    # Initialise ensemble bot population with training data
    population = [
        trader_bots.ensemble_bot(
        ohlcv_df=ohlcv_df,
        all_bot_signals=all_bot_signals,
        all_bot_names=all_bot_names,
        min_terms_in_conjunction=min_terms_in_conjunction,
        max_terms_in_conjunction=max_terms_in_conjunction,
        min_conjunctions_in_dnf=min_conjunctions_in_dnf,
        max_conjunctions_in_dnf=max_conjunctions_in_dnf
    ) for ensemble_bot in range(0, population_size)]

    # maintain a list to hold each instance of the population, its buy_dnf, sell_dnf and final balance
    population_with_info = [] 
    for ensemble_bot in population:
        buy_dnf, sell_dnf, trade_signals = ensemble_bot.generate_signals()
        final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
        ensemble_bot_with_info = [ensemble_bot, buy_dnf, sell_dnf, trade_results, final_balance]
        population_with_info.append(ensemble_bot_with_info)

    # Rank population_with_info by final_balance from highest to lowest
    population_with_info = sorted(population_with_info, key=lambda x: x[4], reverse=True)

    for generation in range(1, number_of_generations + 1):

        # crossover 
        offspring_with_info = crossover(ohlcv_df, fee_percentage, all_bot_signals, all_bot_names, population_with_info)

        ### Perform random mutation of a conjunction for offspring
        for index in range(0, len(offspring_with_info) - 1):
            ensemble_bot = offspring_with_info[index]

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
                    ohlcv_df = ohlcv_df,
                    all_bot_signals = all_bot_signals,
                    all_bot_names = all_bot_names,
                    min_terms_in_conjunction = min_terms_in_conjunction,
                    max_terms_in_conjunction = max_terms_in_conjunction,
                    min_conjunctions_in_dnf = min_conjunctions_in_dnf,
                    max_conjunctions_in_dnf = max_conjunctions_in_dnf
                )

                buy_dnf, sell_dnf, trade_signals = mutated_ensemble_bot.generate_signals(buy_dnf, sell_dnf)
                final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
                mutated_ensemble_bot_with_info = [mutated_ensemble_bot, buy_dnf, sell_dnf, trade_results, final_balance]
                # print(f"mutated: \n{mutated_ensemble_bot_with_info[1]} \n{mutated_ensemble_bot_with_info[2]}\n")

                # Replace the non-mutated offspring bot with the mutated one.
                offspring_with_info.pop(index)
                offspring_with_info.append(mutated_ensemble_bot_with_info)

        # Inject 'injected_population_size' random immigrants
        injected_population = [
            trader_bots.ensemble_bot(
                ohlcv_df = ohlcv_df,
                all_bot_signals = all_bot_signals,
                all_bot_names = all_bot_names,
                min_terms_in_conjunction = min_terms_in_conjunction,
                max_terms_in_conjunction = max_terms_in_conjunction,
                min_conjunctions_in_dnf = min_conjunctions_in_dnf,
                max_conjunctions_in_dnf = max_conjunctions_in_dnf
            ) for ensemble_bot in range(0, injected_population_size)
        ]

        injected_population_with_info = []

        for ensemble_bot in injected_population:
            buy_dnf, sell_dnf, trade_signals = ensemble_bot.generate_signals()
            final_balance, trade_results = utils.execute_trades(trade_signals, fee_percentage)
            ensemble_bot_with_info = [ensemble_bot, buy_dnf, sell_dnf, trade_results, final_balance]
            injected_population_with_info.append(ensemble_bot_with_info)

        # Merge original, offspring, and random injected populations
        merged_population_with_info = population_with_info + offspring_with_info + injected_population_with_info

        # Rank the merged population and reduce it to the specified population size
        merged_population_with_info_sorted = sorted(merged_population_with_info, key=lambda x: x[4], reverse=True)
        population_with_info = merged_population_with_info_sorted[0:population_size]

        ### Print message
        print(f"\nGeneration {generation}:\n")
        print(f"Best ensemble bot's buy_dnf: \n{population_with_info[0][1]}\n")
        print(f"Best ensemble bot's sell_dnf: \n{population_with_info[0][2]}\n")
        print(f"Best ensemble bot's final_balance: {population_with_info[0][4]}\n")

    # # Plot the best ensemble bot from the last generation
    # utils.plot_trading_simulation(
    #     [population_with_info[0][3]], 
    #     ["Ensemble Bot"], 
    #     f"Best Ensemble Bot After {number_of_generations} Generations (Population = {population_size})"
    # )

    # return population_with_info[0][3]
    return population_with_info[0]