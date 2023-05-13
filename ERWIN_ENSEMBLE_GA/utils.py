from ccxt import kraken
import matplotlib.pyplot as plt
from pandas import DataFrame
from trader_bots import MACD_bot, bollinger_bands_bot, RSI_bot, VWAP_bot, stochastic_oscillator_bot, SAR_bot, OBV_trend_following_bot, OBV_trend_reversal_bot, ROC_bot,  Awesome_Oscillator_Bot, ensemble_bot
import re
import random

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


def plot_trading_simulation(trade_results_list, bot_names, title_string):
    # Create a new figure and an axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the title with additional padding
    plt.title(title_string, fontsize=16, color='black', pad=20)

    # Create a color cycle (add more colors if needed)
    colors = ['b', 'g']
    
    # Check if bot_names and trade_results_list have the same length
    if len(trade_results_list) != len(bot_names):
        raise ValueError("trade_results_list and bot_names must have the same length.")

    min_date = min([df.index[0] for df in trade_results_list])
    max_date = max([df.index[-1] for df in trade_results_list])
    
    # Plot each set of trade results
    for i in range(len(trade_results_list)):
        ax.plot(trade_results_list[i].index, trade_results_list[i]["portfolio_value"], color=colors[i % len(colors)], linewidth=2, label=f'{bot_names[i]} Bot')

    # Set grid with less density and make it gray
    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)

    # Set x and y labels with smaller font size and additional padding
    ax.set_xlabel("Day of Trading", fontsize=12, labelpad=10)
    ax.set_ylabel("Portfolio Value in AUD at Close", fontsize=12, labelpad=10)

    # Make the start and end of the x-axis align with the first and last datapoint
    ax.set_xlim([min_date, max_date])

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()

    # Save the figure in high quality at the specified path
    fig.savefig('bot_simulation_results.png', format='png', dpi=300)


def initialise_bots(ohlcv_df, all_parameters):
    all_bot_signals = {}
    all_bot_names = []

    for parameter_list in all_parameters:
        # Get the bot name and remove it from the dictionary.
        parameter_list_copy = dict(parameter_list)
        bot_name = parameter_list_copy.pop('bot_name')
        all_bot_names.append(bot_name)
        # Initialize the bot with its specified parameters and save output signals dataframe.
        signals_df = globals()[bot_name](ohlcv_df, **parameter_list_copy).generate_signals()
        all_bot_signals[bot_name] = signals_df

    return all_bot_signals, all_bot_names



def mutate_dnf(dnf_string, all_bot_names):
    # Find all occurrences of bot names in the string.
    bot_names_in_string = re.findall('|'.join(all_bot_names), dnf_string)

    # If no bot names were found, return the original string.
    if not bot_names_in_string:
        return dnf_string

    # Choose a bot name to replace.
    bot_name_to_replace = random.choice(bot_names_in_string)

    # Choose a replacement bot name.
    bot_name_replacement = random.choice(all_bot_names)

    # Ensure that the replacement bot name is different.
    while bot_name_replacement == bot_name_to_replace:
        bot_name_replacement = random.choice(all_bot_names)

    # Find the indices of all occurrences of the bot name to replace.
    indices = [i for i in range(len(dnf_string)) if dnf_string.startswith(bot_name_to_replace, i)]

    # Choose a random index to replace.
    index_to_replace = random.choice(indices)

    # Replace the chosen occurrence of the bot name with the replacement.
    dnf_string = dnf_string[:index_to_replace] + bot_name_replacement + dnf_string[index_to_replace + len(bot_name_to_replace):]

    return dnf_string


