from ccxt import kraken
from pandas import DataFrame, concat
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt



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

    last_trade = -1

    # For each day:
    for index, row in trade_results.iterrows():
        trading_signal = row["trade_signal"]
        next_day_open_price = row["next_day_open"]
    
        # Records daily portfolio value in AUD at market close.
        if aud_balance == 0: 
            trade_results.at[index, "portfolio_value"] = btc_balance * row["close"]
        else:
            trade_results.at[index, "portfolio_value"] = aud_balance

        # Executes trade at following day's open price if today's data results in trade signal.
        if trading_signal == 1 and last_trade == -1:  # Buy signal
            # Converts all AUD to BTC using the next day's open price and applies percentage fee.
            btc_balance = aud_balance / next_day_open_price * (1 - fee_percentage)
            aud_balance = 0
            last_trade = 1

        elif trading_signal == -1 and last_trade == 1:  # Sell signal
            # Converts all BTC to AUD using the next day's open price and applies percentage fee.
            aud_balance = btc_balance * next_day_open_price * (1 - fee_percentage)
            btc_balance = 0
            last_trade = -1

    # Converts final holdings to AUD using final day's open price if final holdings are in BTC.
    if aud_balance == 0:
        last_close_price = trade_results["next_day_open"].iloc[-1]
        aud_balance = btc_balance * last_close_price * (1 - fee_percentage)
        btc_balance = 0

    return aud_balance, trade_results



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



class MACD_Trading_Bot:
    def __init__(self, slow_window, fast_window, signal_window):
        self.SLOW_WINDOW = slow_window
        self.FAST_WINDOW = fast_window
        self.SIGNAL_WINDOW = signal_window

    def determine_MACD_signals(self, ohlcv_df):
        """ Computes the MACD histogram using the daily close prices. 
            Identifies the buy/sell signals (changes in histogram sign).
            Returns a DataFrame with all the required data for executing the trades.
        """
        trade_signals = ohlcv_df.copy()  # Create a copy of the DataFrame to avoid modifying the original.

        # The MACD histogram is computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes MACD histogram.
        macd_indicator = MACD(close_prices, window_slow=self.SLOW_WINDOW, window_fast=self.FAST_WINDOW, window_sign=self.SIGNAL_WINDOW)
        trade_signals["MACD_histogram"] = macd_indicator.macd_diff()

        # Identifies the buy/sell signals (sign changes in the MACD histogram).
        trade_signals["sign_flag"] = 0
        trade_signals.loc[trade_signals["MACD_histogram"] > 0, "sign_flag"] = 1
        trade_signals.loc[trade_signals["MACD_histogram"] < 0, "sign_flag"] = -1
        trade_signals["trade_signal"] = trade_signals["sign_flag"].diff() / 2
        trade_signals["trade_signal"].fillna(0, inplace=True)
        trade_signals["trade_signal"] = trade_signals["trade_signal"].round().astype(int)

        # Drop the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(columns=["MACD_histogram", "sign_flag"])

        return trade_signals
    


class Bollinger_Bands_Trading_Bot:
    def __init__(self, window, num_standard_deviations):
        self.WINDOW = window
        self.NUM_STANDARD_DEVIATIONS = num_standard_deviations

    def determine_BB_signals(self, ohlcv_df):
        trade_signals = ohlcv_df.copy()  # Create a copy of the DataFrame to avoid modifying the original.

        # The Bollinger Bands are computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes Bollinger Bands indicators.
        bb_indicator = BollingerBands(close_prices, window=self.WINDOW, window_dev=self.NUM_STANDARD_DEVIATIONS)
        trade_signals["buy_signal"] = bb_indicator.bollinger_lband_indicator()
        trade_signals["sell_signal"] = bb_indicator.bollinger_hband_indicator()

        # Combine buy and sell signals into a single trading_signal column.
        trade_signals["trade_signal"] = trade_signals["buy_signal"] - trade_signals["sell_signal"]
        trade_signals["trade_signal"] = trade_signals["trade_signal"].round().astype(int)

        # Drop the unwanted columns from trade_signals
        trade_signals = trade_signals.drop(columns=["buy_signal", "sell_signal"])

        return trade_signals