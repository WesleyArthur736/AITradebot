from ccxt import kraken
import matplotlib.pyplot as plt
from pandas import DataFrame
import random

from trader_bots import MACD_bot, bollinger_bands_bot, RSI_bot, VWAP_bot, stochastic_oscillator_bot, SAR_bot, OBV_trend_following_bot, OBV_trend_reversal_bot, ROC_bot, Awesome_Oscillator_Bot




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


def execute_trades(trade_signals, fee_percentage):
    """ Executes all of the identified trade signals sequentially.
        Ensures the final holdings are in AUD.
        Returns the trading account's final balance in AUD.
    """
    trade_results = trade_signals.copy()
    # trade_results = trade_signals
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

def select_initial_strats(all_strategies, number_of_conjuncts):
    """
    this shall eventually be optimized by the GA
    """

    # randomly select "self.number_of_conjuncts" strategies from "self.all_strategies"
    selected_strats_to_use = random.sample(all_strategies, number_of_conjuncts)

    return selected_strats_to_use

def construct_cnf(trade_type, strategies_to_use):
    # # Chooses the strategies used in the conjunction.
    # number_of_strategies_included = random.randint(self.min_literals, self.max_literals)
    # all_strategies = random.sample(self.strategy_names, number_of_strategies_included)

    # Constructs the conjunction by ANDing the signals from the selected strategies.
    buy_signals = []
    # for strategy_name in self.all_strategies: # basically self.all_strategies is "number_of_conjuncts"
    for strategy_name in strategies_to_use: # basically self.all_strategies is "number_of_conjuncts"
        bot_signals = f"all_bot_signals['{strategy_name}']"
        buy_signal = f"{bot_signals}.at[index, '{trade_type}_signal']"
        buy_signals.append(buy_signal)
    conjunction = " and ".join(buy_signals)
    
    return conjunction

def construct_dnf(trade_type, number_of_disjuncts, strategies_to_use):
    # # Chooses how many conjunctions are used in the DNF.
    # number_of_disjuncts = random.randint(1, 4)

    # Constructs the DNF by generating conjunctions and ORing 
    # them together to make a disjunction of conjunctions.
    conjunctions = []
    for i in range(number_of_disjuncts):
        conjunction = construct_cnf(trade_type, strategies_to_use)
        conjunctions.append(conjunction)
    dnf = " or ".join(conjunctions)
    
    return dnf

def initialise_bots(ohlcv_df, constituent_bot_parameters):

    all_bot_signals = {}

    for parameter_list in constituent_bot_parameters:
        # Get the bot name and remove it from the dictionary.
        # parameter_list_copy = dict(parameter_list)
        parameter_list_copy = parameter_list.copy() ### Changed from the line above

        # print(f"parameter_list_copy:\n{parameter_list_copy}")
        bot_name = parameter_list_copy.pop('bot_name')
        # self.strategy_names.append(bot_name)
        # self.all_strategies.append(bot_name) #### Now taken care of by self.get_all_strategies
        # Initialize the bot with its specified parameters and save output signals dataframe.
        # signals_df = globals()[bot_name](ohlcv_df, **parameter_list_copy).generate_signals()

        signals_df = globals()[bot_name](ohlcv_df, **parameter_list_copy).generate_signals()

        # signals_df, buy_dnf, sell_dnf = type(bot_name)(
        #     ohlcv_df,
        #     **parameter_list
        # ).generate_signals()

        all_bot_signals[bot_name] = signals_df

    return all_bot_signals

def mutate_dnf(dnf, is_buy_dnf, all_strategies, num_disjuncts_mutate = 0):
    """
    Example usage/effect of applying this function:

    dnf           = (A and B and C) or (A and D and E) or (B and E and F)
    mutated_dnf   = mutate_dnf(dnf, all_strategies)
    mutated_dnf   = (A and B and C) or (F and D and E) or (B and E and F)

    mutated_dnf_2 = mutate_dnf(dnf, all_strategies, num_disjuncts_mutate = 2)
    mutated_dnf_2 = (A and B and C) or (F and D and E) or (B and E and F) or H or (A and G and H) or (I and J and K)
    """

    if is_buy_dnf == True:
        signal_type = "buy_signal"
    else:
        signal_type = "sell_signal"

    # Split the DNF into a list of conjunctions
    conjunctions = dnf.split(" or ")
    
    # Select a random conjunction to mutate
    mutation_idx = random.randint(0, len(conjunctions) - 1)

    mutated_conjunction = conjunctions[mutation_idx]
    
    # Split the conjunction into a list of conditions
    conditions = mutated_conjunction.split(" and ")
    
    # Select a random condition to mutate
    condition_idx = random.randint(0, len(conditions) - 1)
    mutated_condition = conditions[condition_idx]

    # print(f"mutated_condition: {mutated_condition}")
    # print(f'mutated_condition.split(".")[1][10:]: {mutated_condition.split(".")[1][10:]}')
    
    # Get the current index of the condition in all_bot_signals
    current_idx = all_strategies.index(mutated_condition.split("[")[1][1:-5])

    # Select a new index that is different from the current one
    new_idx = random.randint(0, len(all_strategies) - 1)
    # while new_idx == current_idx:
    while new_idx == current_idx:

        new_idx = random.randint(0, len(all_strategies) - 1)

        while all_strategies[current_idx] == all_strategies[new_idx]:
            new_idx = random.randint(0, len(all_strategies) - 1)
    
    # Replace the old index with the new one in the mutated condition
    mutated_condition = mutated_condition.replace(all_strategies[current_idx], all_strategies[new_idx])
    
    # Replace the old condition with the mutated one in the mutated conjunction
    conditions[condition_idx] = mutated_condition
    mutated_conjunction = " and ".join(conditions)
    
    # Replace the old conjunction with the mutated one in the original DNF
    conjunctions[mutation_idx] = mutated_conjunction
    
    # Add/subtract num_disjuncts_mutate new disjuncts to the mutated DNF
    if num_disjuncts_mutate > 0:
        new_disjuncts = random.sample(all_strategies, num_disjuncts_mutate)

        new_disjuncts = [f"all_bot_signals['{new_disjunct}'].at[index, '{signal_type}']" for new_disjunct in new_disjuncts]

        new_conjunctions = [" and ".join(random.sample(new_disjuncts, random.randint(1, len(new_disjuncts)))) for _ in range(num_disjuncts_mutate)]

        for new_conjunction in new_conjunctions:
            while new_conjunctions in conjunctions:
                new_conjunctions = [" and ".join(random.sample(new_disjuncts, random.randint(1, len(new_disjuncts)))) for _ in range(num_disjuncts_mutate)]
        conjunctions.extend(new_conjunctions)

    elif num_disjuncts_mutate < 0:
        num_disjuncts_mutate = min(abs(num_disjuncts_mutate), len(conjunctions))
        del conjunctions[-num_disjuncts_mutate:]
    
    mutated_dnf = " or ".join(conjunctions)
    
    return mutated_dnf

# def mutate_dnf(dnf, all_strategies):
#     """
#     Example usage/effect of applying this function:
    
#     dnf         = (A and B and C) or (A and D and E) or (B and E and F)
#     mutated_dnf = mutate_dnf(dnf)
#     mutated_dnf = (A and B and C) or (F and D and E) or (B and E and F)
#     """

#     # Split the DNF into a list of conjunctions
#     conjunctions = dnf.split(" or ")
    
#     # Select a random conjunction to mutate
#     mutation_idx = random.randint(0, len(conjunctions) - 1)

#     mutated_conjunction = conjunctions[mutation_idx]
    
#     # Split the conjunction into a list of conditions
#     conditions = mutated_conjunction.split(" and ")
    
#     # Select a random condition to mutate
#     condition_idx = random.randint(0, len(conditions) - 1)
#     mutated_condition = conditions[condition_idx]
    
#     # Get the current index of the condition in all_bot_signals
#     current_idx = all_strategies.index(mutated_condition.split("[")[1][1:-5])

#     # Select a new index that is different from the current one
#     new_idx = random.randint(0, len(all_strategies) - 1)
#     # while new_idx == current_idx:
#     while new_idx == current_idx:

#         new_idx = random.randint(0, len(all_strategies) - 1)

#         while all_strategies[current_idx] == all_strategies[new_idx]:
#             new_idx = random.randint(0, len(all_strategies) - 1)
    
#     # Replace the old index with the new one in the mutated condition
#     mutated_condition = mutated_condition.replace(all_strategies[current_idx], all_strategies[new_idx])
    
#     # Replace the old condition with the mutated one in the mutated conjunction
#     conditions[condition_idx] = mutated_condition
#     mutated_conjunction = " and ".join(conditions)
    
#     # Replace the old conjunction with the mutated one in the original DNF
#     conjunctions[mutation_idx] = mutated_conjunction
#     mutated_dnf = " or ".join(conjunctions)
    
#     return mutated_dnf


# def mutate_dnf(dnf, all_strategies):
#     """
#     Example usage/effect of applying this function:

#     dnf         = (A and B and C) or (A and D and E) or (B and E and F)
#     mutated_dnf = mutate_dnf(dnf)
#     mutated_dnf = (A and B and C) or (F and D and E) or (B and E and F)
#     """

#     # Split the DNF into a list of conjunctions
#     conjunctions = dnf.split(" or ")
    
#     # Select a random conjunction to mutate
#     mutation_idx = random.randint(0, len(conjunctions) - 1)

#     mutated_conjunction = conjunctions[mutation_idx]
    
#     # Split the conjunction into a list of conditions
#     conditions = mutated_conjunction.split(" and ")
    
#     # Select a random condition to mutate
#     condition_idx = random.randint(0, len(conditions) - 1)
#     mutated_condition = conditions[condition_idx]
    
#     # Get the current index of the condition in all_bot_signals
#     current_idx = all_strategies.index(mutated_condition.split("[")[1][1:-5])

#     # Select a new index that is different from the current one
#     new_idx = random.randint(0, len(all_strategies) - 1)
#     # while new_idx == current_idx:
#     while new_idx == current_idx:

#         new_idx = random.randint(0, len(all_strategies) - 1)

#         while all_strategies[current_idx] == all_strategies[new_idx]:
#             new_idx = random.randint(0, len(all_strategies) - 1)
    
#     # Replace the old index with the new one in the mutated condition
#     mutated_condition = mutated_condition.replace(all_strategies[current_idx], all_strategies[new_idx])
    
#     # Replace the old condition with the mutated one in the mutated conjunction
#     conditions[condition_idx] = mutated_condition
#     mutated_conjunction = " and ".join(conditions)
    
#     # Replace the old conjunction with the mutated one in the original DNF
#     conjunctions[mutation_idx] = mutated_conjunction
#     mutated_dnf = " or ".join(conjunctions)
    
#     return mutated_dnf

def plot_trading_simulation(trade_results, bot_type):
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
    ax.set_title(f"{bot_type} Bot Simulation Results")

    # Display the plot
    plt.show()



# def construct_cnf(self, trade_type):
    #     # # Chooses the strategies used in the conjunction.
    #     # number_of_strategies_included = random.randint(self.min_literals, self.max_literals)
    #     # all_strategies = random.sample(self.strategy_names, number_of_strategies_included)

    #     # Constructs the conjunction by ANDing the signals from the selected strategies.
    #     buy_signals = []
    #     # for strategy_name in self.all_strategies: # basically self.all_strategies is "number_of_conjuncts"
    #     for strategy_name in self.strategies_to_use: # basically self.all_strategies is "number_of_conjuncts"
    #         bot_signals = f"all_bot_signals['{strategy_name}']"
    #         buy_signal = f"{bot_signals}.at[index, '{trade_type}_signal']"
    #         buy_signals.append(buy_signal)
    #     conjunction = " and ".join(buy_signals)
        
    #     return conjunction

    # def construct_dnf(self, trade_type):
    #     # # Chooses how many conjunctions are used in the DNF.
    #     # number_of_disjuncts = random.randint(1, 4)

    #     # Constructs the DNF by generating conjunctions and ORing 
    #     # them together to make a disjunction of conjunctions.
    #     conjunctions = []
    #     for i in range(self.number_of_disjuncts):
    #         conjunction = self.construct_cnf(trade_type)
    #         conjunctions.append(conjunction)
    #     dnf = " or ".join(conjunctions)
        
    #     return dnf

