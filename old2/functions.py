from ccxt import kraken
from pandas import DataFrame, concat
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import StochasticOscillator
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


class Bots:
    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df

    def MACD_bot(self, slow_window, fast_window, signal_window):
        """ Computes the MACD histogram using the daily close prices. 
            Identifies the buy/sell signals (changes in histogram sign).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        
        # The MACD histogram is computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes MACD histogram.
        macd_indicator = MACD(close_prices, window_slow = slow_window, window_fast = fast_window, window_sign = signal_window)
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
            
            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = MACD_histogram_was_negative and MACD_histogram_now_positive
            sell_signal = MACD_histpgram_was_positive and MACD_histogram_now_negative

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns = ["MACD_histogram"])

        return trade_signals
    

    def bollinger_bands_bot(self, window, num_standard_deviations):
        """ Computes the Bollinger band values using the daily close prices.
            Identifies the buy/sell signals (price exiting the bands).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()  

        # The Bollinger Bands are computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes Bollinger Bands indicators.
        bb_indicator = BollingerBands(close_prices, window = window, window_dev = num_standard_deviations)
        trade_signals["BB_highband"] = bb_indicator.bollinger_hband()
        trade_signals["BB_lowband"] = bb_indicator.bollinger_lband()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iterrows():
            # Evaluates literals. 
            price_above_BB_highband = trade_signals.at[index, "close"] > trade_signals.at[index, "BB_highband"]
            price_below_BB_lowband = trade_signals.at[index, "BB_lowband"] > trade_signals.at[index, "close"]
            
            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = price_below_BB_lowband
            sell_signal = price_above_BB_highband

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals
        trade_signals = trade_signals.drop(columns=["BB_highband", "BB_lowband"])

        return trade_signals
    

    def RSI_bot(self, overbought_threshold, oversold_threshold, window):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        
        # The RSI values are computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes RSI values.
        rsi_indicator = RSIIndicator(close_prices, window = window)
        trade_signals['rsi'] = rsi_indicator.rsi()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iterrows():
            # Evaluates literals.
            rsi_above_overbought_threshold = trade_signals.at[index, 'rsi'] > overbought_threshold
            rsi_below_oversold_threshold = oversold_threshold > trade_signals.at[index, 'rsi']

            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = rsi_below_oversold_threshold
            sell_signal = rsi_above_overbought_threshold

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(columns=['rsi'])

        return trade_signals


    def VWAP_bot(self, window):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        
        # The VWAP values are computed from the high, low, close and volume data.
        vwap_indicator = VolumeWeightedAveragePrice(
            high = trade_signals['high'],
            low = trade_signals['low'],
            close = trade_signals['close'],
            volume = trade_signals['volume'],
            window = window 
        )
        trade_signals['vwap'] = vwap_indicator.volume_weighted_average_price()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iterrows():
            # Evaluates literals.
            price_above_vwap = trade_signals.at[index, 'close'] > trade_signals.at[index, 'vwap']
            price_below_vwap = trade_signals.at[index, 'vwap'] > trade_signals.at[index, 'close']

            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = price_below_vwap
            sell_signal = price_above_vwap

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(columns=['vwap'])

        return trade_signals
    

    def stochastic_oscillator_bot(self, oscillator_window, signal_window, overbought_threshold, oversold_threshold):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The stochastic oscillator values are computed from the high, low, and close prices.
        StochOsc = StochasticOscillator(
            trade_signals['close'],
            trade_signals['high'],
            trade_signals['low'],
            window = oscillator_window,
            smooth_window = signal_window
        )
        trade_signals["stoch_oscillator"] = StochOsc.stoch()
        trade_signals["stoch_signal"] = StochOsc.stoch_signal()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            stoch_oscillator_oversold = oversold_threshold > trade_signals.at[index, "stoch_oscillator"]
            stoch_signal_oversold = oversold_threshold > trade_signals.at[index, "stoch_signal"]
            oscillator_was_above_signal = trade_signals.at[index - 1, "stoch_oscillator"] > trade_signals.at[index - 1, "stoch_signal"]
            oscillator_now_below_signal = trade_signals.at[index, "stoch_signal"] > trade_signals.at[index, "stoch_oscillator"]

            stoch_oscillator_overbought = trade_signals.at[index, "stoch_oscillator"] > overbought_threshold
            stoch_signal_overbought = trade_signals.at[index, "stoch_signal"] > overbought_threshold
            oscillator_was_below_signal = trade_signals.at[index - 1, "stoch_signal"] > trade_signals.at[index - 1, "stoch_oscillator"]
            oscillator_now_above_signal = trade_signals.at[index, "stoch_oscillator"] > trade_signals.at[index, "stoch_signal"]

            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = stoch_oscillator_oversold and stoch_signal_oversold and oscillator_was_above_signal and oscillator_now_below_signal
            sell_signal = stoch_oscillator_overbought and stoch_signal_overbought and oscillator_was_below_signal and oscillator_now_above_signal

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(columns=["stoch_oscillator", "stoch_signal"])

        return trade_signals




    
