from ccxt import kraken
from pandas import DataFrame, concat
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import StochasticOscillator
from ta.trend import PSARIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import ROCIndicator
from ta.momentum import AwesomeOscillatorIndicator
import matplotlib.pyplot as plt
import random


class Bot(object):

    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df

    def generate_signals(self):
        """
        This method generates the trading signals based on the bot's technical indicator and input data.
        """
        raise NotImplementedError("Subclass must implement generate_signals method.")


class MACD_bot(Bot):

    def __init__(self, ohlcv_df, slow_window, fast_window, signal_window):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.slow_window = slow_window
        self.fast_window = fast_window
        self.signal_window = signal_window

    def generate_signals(self):
        """ Computes the MACD histogram using the daily close prices. 
            Identifies the buy/sell signals (changes in histogram sign).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        
        # The MACD histogram is computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes MACD histogram.
        macd_indicator = MACD(
            close = close_prices, 
            window_slow = self.slow_window, 
            window_fast = self.fast_window, 
            window_sign = self.signal_window
        )

        # Computes indicator values.
        trade_signals["MACD_histogram"] = macd_indicator.macd_diff()

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


class bollinger_bands_bot(Bot):

    def __init__(self, ohlcv_df, window, num_standard_deviations):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.window = window
        self.num_standard_deviations = num_standard_deviations

    def generate_signals(self):
        """ Computes the Bollinger band values using the daily close prices.
            Identifies the buy/sell signals (price exiting the bands).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()  

        # The Bollinger Bands are computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes Bollinger Bands indicators.
        bb_indicator = BollingerBands(
            close = close_prices, 
            window = self.window, 
            window_dev = self.num_standard_deviations
        )
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


class RSI_bot(Bot):
    
    def __init__(self, ohlcv_df, overbought_threshold, oversold_threshold, window):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.window = window

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        
        # The RSI values are computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes RSI values.
        rsi_indicator = RSIIndicator(
            close = close_prices, 
            window = self.window
        )
        trade_signals['rsi'] = rsi_indicator.rsi()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iterrows():
            # Evaluates literals.
            rsi_above_overbought_threshold = trade_signals.at[index, 'rsi'] > self.overbought_threshold
            rsi_below_oversold_threshold = self.oversold_threshold > trade_signals.at[index, 'rsi']

            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = rsi_below_oversold_threshold
            sell_signal = rsi_above_overbought_threshold

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(columns=['rsi'])

        return trade_signals


class VWAP_bot(Bot):

    def __init__(self, ohlcv_df, window):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.window = window

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        
        # The VWAP values are computed from the high, low, close and volume data.
        vwap_indicator = VolumeWeightedAveragePrice(
            high = trade_signals['high'],
            low = trade_signals['low'],
            close = trade_signals['close'],
            volume = trade_signals['volume'],
            window = self.window 
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


class stochastic_oscillator_bot(Bot):

    def __init__(self, ohlcv_df, oscillator_window, signal_window, overbought_threshold, oversold_threshold):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.oscillator_window = oscillator_window
        self.signal_window = signal_window
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold  = oversold_threshold

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The stochastic oscillator values are computed from the high, low, and close prices.
        StochOsc = StochasticOscillator(
            high = trade_signals['high'],
            low = trade_signals['low'],
            close = trade_signals['close'],
            window = self.oscillator_window,
            smooth_window = self.signal_window
        )

        trade_signals["stoch_oscillator"] = StochOsc.stoch()
        trade_signals["stoch_signal"] = StochOsc.stoch_signal()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            stoch_oscillator_oversold = self.oversold_threshold > trade_signals.at[index, "stoch_oscillator"]
            stoch_signal_oversold = self.oversold_threshold > trade_signals.at[index, "stoch_signal"]
            oscillator_was_above_signal = trade_signals.at[index - 1, "stoch_oscillator"] > trade_signals.at[index - 1, "stoch_signal"]
            oscillator_now_below_signal = trade_signals.at[index, "stoch_signal"] > trade_signals.at[index, "stoch_oscillator"]

            stoch_oscillator_overbought = trade_signals.at[index, "stoch_oscillator"] > self.overbought_threshold
            stoch_signal_overbought = trade_signals.at[index, "stoch_signal"] > self.overbought_threshold
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
    

class SAR_bot(Bot):

    def __init__(self, ohlcv_df, step, max_step):
        super().__init__(ohlcv_df)
        self.step = step
        self.max_step = max_step

    def generate_signals(self):
        """ Computes the Parabolic SAR using the daily high and low prices.
            Identifies the buy/sell signals (changes in SAR direction).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The Parabolic SAR is computed from the daily high and low prices.
        high_prices = trade_signals["high"]
        low_prices = trade_signals["low"]
        close_prices = trade_signals["close"]

        # Computes Parabolic SAR indicator.
        sar_indicator = PSARIndicator(
            high = high_prices, 
            low = low_prices, 
            close = close_prices,
            step = self.step, 
            max_step = self.max_step
        )

        # Computes indicator values.
        trade_signals["SAR"] = sar_indicator.psar()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals. 
            SAR_was_below_price = trade_signals.at[index - 1, "close"] > trade_signals.at[index - 1, "SAR"] 
            SAR_was_above_price = trade_signals.at[index - 1, "SAR"] > trade_signals.at[index - 1, "close"]
            SAR_is_below_price = trade_signals.at[index, "close"] > trade_signals.at[index, "SAR"]
            SAR_is_above_price = trade_signals.at[index, "SAR"] > trade_signals.at[index, "close"]
            
            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = SAR_was_below_price and SAR_is_above_price
            sell_signal = SAR_was_above_price and SAR_is_below_price

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns = ["SAR"])

        return trade_signals


class OBV_trend_following_bot(Bot):

    def __init__(self, ohlcv_df):
        super().__init__(ohlcv_df)

    def generate_signals(self):
        """ Computes the On-Balance Volume (OBV) using the daily close prices and volume.
            Identifies the buy/sell signals (price/OBV rising or price/OBV falling).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The OBV is computed from the daily close prices and volume.
        close_prices = trade_signals["close"]
        volumes = trade_signals["volume"]

        # Computes OBV indicator.
        obv_indicator = OnBalanceVolumeIndicator(
            close = close_prices, 
            volume = volumes
        )

        # Computes indicator values.
        trade_signals["OBV"] = obv_indicator.on_balance_volume()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals. 
            OBV_rising = trade_signals.at[index, "OBV"] > trade_signals.at[index - 1, "OBV"]
            price_rising = trade_signals.at[index, "close"] > trade_signals.at[index - 1, "close"]
            OBV_falling = trade_signals.at[index - 1, "OBV"] > trade_signals.at[index, "OBV"]
            price_falling = trade_signals.at[index - 1, "close"] > trade_signals.at[index, "close"]
            
            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = OBV_rising and price_rising
            sell_signal = OBV_falling and price_falling

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns = ["OBV"])

        return trade_signals


class OBV_trend_reversal_bot(Bot):

    def __init__(self, ohlcv_df):
        super().__init__(ohlcv_df)

    def generate_signals(self):
        """ Computes the On-Balance Volume (OBV) using the daily close prices and volume.
            Identifies the buy/sell signals (rising price and falling OBV or vice versa).
            Returns a DataFrame with all the required data for executing the trades.
        """
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The OBV is computed from the daily close prices and volume.
        close_prices = trade_signals["close"]
        volumes = trade_signals["volume"]

        # Computes OBV indicator.
        obv_indicator = OnBalanceVolumeIndicator(
            close = close_prices, 
            volume = volumes
        )

        # Computes indicator values.
        trade_signals["OBV"] = obv_indicator.on_balance_volume()

        # Initialises output columns.
        trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals. 
            OBV_rising = trade_signals.at[index, "OBV"] > trade_signals.at[index - 1, "OBV"]
            price_rising = trade_signals.at[index, "close"] > trade_signals.at[index - 1, "close"]
            OBV_falling = trade_signals.at[index - 1, "OBV"] > trade_signals.at[index, "OBV"]
            price_falling = trade_signals.at[index - 1, "close"] > trade_signals.at[index, "close"]
            
            # Evaluates buy and sell conjunctions to determine buy and sell signals. 
            buy_signal = price_falling and OBV_rising
            sell_signal = price_rising and OBV_falling

            # Records buy and sell signals. 
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns = ["OBV"])

        return trade_signals


class ROC_bot(Bot):
    def __init__(self, ohlcv_df, window, buy_threshold, sell_threshold):
        super().__init__(ohlcv_df)
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # Compute the ROC indicator.
        roc = ROCIndicator(close=trade_signals["close"], window=self.window)
        trade_signals["roc"] = roc.roc()

        # Initialize output columns.
        trade_signals["buy_signal"] = False    # Initialize output column for the buy signals.
        trade_signals["sell_signal"] = False   # Initialize output column for the sell signals.

        for index, row in trade_signals.iloc[self.window + 1:].iterrows():
            # Check if the ROC indicator crossed the buy/sell threshold.
            roc_above_buy_threshold = trade_signals.at[index, "roc"] > self.buy_threshold
            roc_below_sell_threshold = trade_signals.at[index, "roc"] < self.sell_threshold

            # Determine buy and sell signals.
            buy_signal = roc_above_buy_threshold
            sell_signal = roc_below_sell_threshold

            # Record buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drop the ROC indicator.
        trade_signals = trade_signals.drop(columns=["roc"])

        return trade_signals


class Awesome_Oscillator_Bot(Bot):
    def __init__(self, ohlcv_df, window1, window2):
        super().__init__(ohlcv_df)
        self.ohlcv_df = ohlcv_df
        self.window1 = window1
        self.window2 = window2

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # Get the High and Low Prices
        high_price = trade_signals['high']
        low_price = trade_signals['low']

        # Compute the Awesome Indicator
        ao_indicator = AwesomeOscillatorIndicator(
            high=high_price,
            low=low_price,
            window1=self.window1,
            window2=self.window2
        )
        trade_signals['ao'] = ao_indicator.awesome_oscillator()

        # Initialises output columns.
        trade_signals["buy_signal"] = False # Initialises output column for the buy signals.
        trade_signals["sell_signal"] = False # Initialises output column for the sell signals.

        for index, row in trade_signals.iterrows():
            # Evaluate literals
            awesome_above_zero = trade_signals.at[index, 'ao'] > 0 # buying pressure
            awesome_below_zero = 0 > trade_signals.at[index, 'ao'] # selling pressure

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = awesome_above_zero
            sell_signal = awesome_below_zero

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(columns=['ao'])

        return trade_signals





class ensemble_bot(Bot):

    def __init__(self, ohlcv_df, all_bot_signals, all_bot_names, min_terms_in_conjunction, max_terms_in_conjunction, min_conjunctions_in_dnf, max_conjunctions_in_dnf):
        super().__init__(ohlcv_df)
        self.all_bot_signals = all_bot_signals
        self.all_bot_names = all_bot_names
        self.min_terms_in_conjunction = min_terms_in_conjunction
        self.max_terms_in_conjunction = max_terms_in_conjunction
        self.min_conjunctions_in_dnf = min_conjunctions_in_dnf
        self.max_conjunctions_in_dnf = max_conjunctions_in_dnf
    

    def construct_conjunction(self, trade_type):
        # Chooses the strategies used in the conjunction.
        number_of_bots_included = random.randint(self.min_terms_in_conjunction, self.max_terms_in_conjunction)
        bots_used = random.sample(self.all_bot_names, number_of_bots_included)

        # Constructs the conjunction by ANDing the signals from the selected strategies.
        buy_signals = []
        for bot in bots_used:
            bot_signals = f"all_bot_signals['{bot}']"
            buy_signal = f"{bot_signals}.at[index, '{trade_type}_signal']"
            buy_signals.append(buy_signal)
        conjunction = " and ".join(buy_signals)
        
        return conjunction

    def construct_dnf(self, trade_type):
        # Chooses how many conjunctions are used in the DNF.
        number_of_conjunctions = random.randint(self.min_conjunctions_in_dnf, self.max_conjunctions_in_dnf)

        # Constructs the DNF by generating conjunctions and ORing them together.
        conjunctions = []
        for i in range(number_of_conjunctions):
                conjunction = self.construct_conjunction(trade_type)
                conjunctions.append(conjunction)
        dnf = " or ".join(conjunctions)
        
        return dnf

    def generate_signals(self, buy_dnf = False, sell_dnf = False):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()
        all_bot_signals = self.all_bot_signals

        # Generate a random buy_dnf and sell_dnf if not provided.
        if buy_dnf == False or sell_dnf == False: 
            buy_dnf = self.construct_dnf(trade_type = "buy")
            sell_dnf = self.construct_dnf(trade_type = "sell")

        # Evaluate DNF expression for each day of data and save to dataframe.
        for index, row in trade_signals.iterrows():
            buy_dnf_with_index = buy_dnf.replace("index", str(index))
            buy_signal = eval(buy_dnf_with_index)
            trade_signals.at[index, "buy_signal"] = buy_signal 

        # Evaluate DNF expression for each day of data and save to dataframe.
        for index, row in trade_signals.iterrows(): 
            sell_dnf_with_index = sell_dnf.replace("index", str(index))
            sell_signal = eval(sell_dnf_with_index)
            trade_signals.at[index, "sell_signal"] = sell_signal 

        return buy_dnf, sell_dnf, trade_signals


