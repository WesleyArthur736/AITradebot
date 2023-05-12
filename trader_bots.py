from ccxt import kraken
from pandas import DataFrame, concat
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import StochasticOscillator
from ta.trend import PSARIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import ROCIndicator, AwesomeOscillatorIndicator
import matplotlib.pyplot as plt
import random


class Bot(object):

    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df

    def generate_signals(self):
        """
        This method generates the trading signals based on the bot's technical indicator and input data.
        """
        raise NotImplementedError(
            "Subclass must implement generate_signals method.")


class MACD_bot(Bot):

    def __init__(self, ohlcv_df, slow_window, fast_window, signal_window):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.slow_window = slow_window
        self.fast_window = fast_window
        self.signal_window = signal_window

        self.bot_type = "MACD"
        self.params = [slow_window, fast_window, signal_window]

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
            close=close_prices,
            window_slow=self.slow_window,
            window_fast=self.fast_window,
            window_sign=self.signal_window
        )

        # Computes indicator values.
        trade_signals["MACD_histogram"] = macd_indicator.macd_diff()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            MACD_histogram_was_negative = 0 > trade_signals.at[index -
                                                               1, "MACD_histogram"]
            MACD_histpgram_was_positive = trade_signals.at[index -
                                                           1, "MACD_histogram"] > 0
            MACD_histogram_now_negative = 0 > trade_signals.at[index,
                                                               "MACD_histogram"]
            MACD_histogram_now_positive = trade_signals.at[index,
                                                           "MACD_histogram"] > 0

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = MACD_histogram_was_negative and MACD_histogram_now_positive
            sell_signal = MACD_histpgram_was_positive and MACD_histogram_now_negative

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns=["MACD_histogram"])

        return trade_signals


class bollinger_bands_bot(Bot):

    def __init__(self, ohlcv_df, window, num_standard_deviations):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.window = window
        self.num_standard_deviations = num_standard_deviations

        self.bot_type = "Bollinger"
        self.params = [window, num_standard_deviations]

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
            close=close_prices,
            window=self.window,
            window_dev=self.num_standard_deviations
        )
        trade_signals["BB_highband"] = bb_indicator.bollinger_hband()
        trade_signals["BB_lowband"] = bb_indicator.bollinger_lband()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iterrows():
            # Evaluates literals.
            price_above_BB_highband = trade_signals.at[index,
                                                       "close"] > trade_signals.at[index, "BB_highband"]
            price_below_BB_lowband = trade_signals.at[index,
                                                      "BB_lowband"] > trade_signals.at[index, "close"]

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = price_below_BB_lowband
            sell_signal = price_above_BB_highband

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals
        trade_signals = trade_signals.drop(
            columns=["BB_highband", "BB_lowband"])

        return trade_signals


class RSI_bot(Bot):

    def __init__(self, ohlcv_df, overbought_threshold, oversold_threshold, window):
        # super().__init__(ohlcv_df, trade_signals, close_prices)
        super().__init__(ohlcv_df)
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.window = window

        self.bot_type = "RSI"
        self.params = [overbought_threshold, oversold_threshold, window]

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The RSI values are computed from the daily close prices.
        close_prices = trade_signals["close"]

        # Computes RSI values.
        rsi_indicator = RSIIndicator(
            close=close_prices,
            window=self.window
        )
        trade_signals['rsi'] = rsi_indicator.rsi()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iterrows():
            # Evaluates literals.
            rsi_above_overbought_threshold = trade_signals.at[index,
                                                              'rsi'] > self.overbought_threshold
            rsi_below_oversold_threshold = self.oversold_threshold > trade_signals.at[
                index, 'rsi']

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

        self.bot_type = "VWAP"
        self.params = [window]

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The VWAP values are computed from the high, low, close and volume data.
        vwap_indicator = VolumeWeightedAveragePrice(
            high=trade_signals['high'],
            low=trade_signals['low'],
            close=trade_signals['close'],
            volume=trade_signals['volume'],
            window=self.window
        )
        trade_signals['vwap'] = vwap_indicator.volume_weighted_average_price()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iterrows():
            # Evaluates literals.
            price_above_vwap = trade_signals.at[index,
                                                'close'] > trade_signals.at[index, 'vwap']
            price_below_vwap = trade_signals.at[index,
                                                'vwap'] > trade_signals.at[index, 'close']

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
        self.oversold_threshold = oversold_threshold

        self.bot_type = "Stochastic Oscillator"
        self.params = [oscillator_window, signal_window,
                       overbought_threshold, oversold_threshold]

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # The stochastic oscillator values are computed from the high, low, and close prices.
        StochOsc = StochasticOscillator(
            high=trade_signals['high'],
            low=trade_signals['low'],
            close=trade_signals['close'],
            window=self.oscillator_window,
            smooth_window=self.signal_window
        )

        trade_signals["stoch_oscillator"] = StochOsc.stoch()
        trade_signals["stoch_signal"] = StochOsc.stoch_signal()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            stoch_oscillator_oversold = self.oversold_threshold > trade_signals.at[
                index, "stoch_oscillator"]
            stoch_signal_oversold = self.oversold_threshold > trade_signals.at[
                index, "stoch_signal"]
            oscillator_was_above_signal = trade_signals.at[index - 1,
                                                           "stoch_oscillator"] > trade_signals.at[index - 1, "stoch_signal"]
            oscillator_now_below_signal = trade_signals.at[index,
                                                           "stoch_signal"] > trade_signals.at[index, "stoch_oscillator"]

            stoch_oscillator_overbought = trade_signals.at[index,
                                                           "stoch_oscillator"] > self.overbought_threshold
            stoch_signal_overbought = trade_signals.at[index,
                                                       "stoch_signal"] > self.overbought_threshold
            oscillator_was_below_signal = trade_signals.at[index - 1,
                                                           "stoch_signal"] > trade_signals.at[index - 1, "stoch_oscillator"]
            oscillator_now_above_signal = trade_signals.at[index,
                                                           "stoch_oscillator"] > trade_signals.at[index, "stoch_signal"]

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = stoch_oscillator_oversold and stoch_signal_oversold and oscillator_was_above_signal and oscillator_now_below_signal
            sell_signal = stoch_oscillator_overbought and stoch_signal_overbought and oscillator_was_below_signal and oscillator_now_above_signal

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted columns from trade_signals.
        trade_signals = trade_signals.drop(
            columns=["stoch_oscillator", "stoch_signal"])

        return trade_signals


class SAR_bot(Bot):

    def __init__(self, ohlcv_df, step, max_step):
        super().__init__(ohlcv_df)
        self.step = step
        self.max_step = max_step

        self.bot_type = "SAR"
        self.params = [step, max_step]

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
            high=high_prices,
            low=low_prices,
            close=close_prices,
            step=self.step,
            max_step=self.max_step
        )

        # Computes indicator values.
        trade_signals["SAR"] = sar_indicator.psar()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            SAR_was_below_price = trade_signals.at[index - 1,
                                                   "close"] > trade_signals.at[index - 1, "SAR"]
            SAR_was_above_price = trade_signals.at[index - 1,
                                                   "SAR"] > trade_signals.at[index - 1, "close"]
            SAR_is_below_price = trade_signals.at[index,
                                                  "close"] > trade_signals.at[index, "SAR"]
            SAR_is_above_price = trade_signals.at[index,
                                                  "SAR"] > trade_signals.at[index, "close"]

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = SAR_was_below_price and SAR_is_above_price
            sell_signal = SAR_was_above_price and SAR_is_below_price

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns=["SAR"])

        return trade_signals


class OBV_trend_following_bot(Bot):

    def __init__(self, ohlcv_df):
        super().__init__(ohlcv_df)
        self.bot_type = "OVB Trend Following"
        self.params = []

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
            close=close_prices,
            volume=volumes
        )

        # Computes indicator values.
        trade_signals["OBV"] = obv_indicator.on_balance_volume()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            OBV_rising = trade_signals.at[index,
                                          "OBV"] > trade_signals.at[index - 1, "OBV"]
            price_rising = trade_signals.at[index,
                                            "close"] > trade_signals.at[index - 1, "close"]
            OBV_falling = trade_signals.at[index - 1,
                                           "OBV"] > trade_signals.at[index, "OBV"]
            price_falling = trade_signals.at[index - 1,
                                             "close"] > trade_signals.at[index, "close"]

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = OBV_rising and price_rising
            sell_signal = OBV_falling and price_falling

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns=["OBV"])

        return trade_signals


class OBV_trend_reversal_bot(Bot):

    def __init__(self, ohlcv_df):
        super().__init__(ohlcv_df)
        self.bot_type = "OVB Trend Reversal"
        self.params = []

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
            close=close_prices,
            volume=volumes
        )

        # Computes indicator values.
        trade_signals["OBV"] = obv_indicator.on_balance_volume()

        # Initialises output columns.
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iloc[1:].iterrows():
            # Evaluates literals.
            OBV_rising = trade_signals.at[index,
                                          "OBV"] > trade_signals.at[index - 1, "OBV"]
            price_rising = trade_signals.at[index,
                                            "close"] > trade_signals.at[index - 1, "close"]
            OBV_falling = trade_signals.at[index - 1,
                                           "OBV"] > trade_signals.at[index, "OBV"]
            price_falling = trade_signals.at[index - 1,
                                             "close"] > trade_signals.at[index, "close"]

            # Evaluates buy and sell conjunctions to determine buy and sell signals.
            buy_signal = price_falling and OBV_rising
            sell_signal = price_rising and OBV_falling

            # Records buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drops the unwanted column from output dataframe.
        trade_signals = trade_signals.drop(columns=["OBV"])

        return trade_signals


class ROC_bot(Bot):
    def __init__(self, ohlcv_df, window, buy_threshold, sell_threshold):
        super().__init__(ohlcv_df)
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.bot_type = "ROC"
        self.params = [window, buy_threshold, sell_threshold]

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        # Compute the ROC indicator.
        roc = ROCIndicator(close=trade_signals["close"], window=self.window)
        trade_signals["roc"] = roc.roc()

        # Initialize output columns.
        # Initialize output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialize output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iloc[self.window + 1:].iterrows():
            # Check if the ROC indicator crossed the buy/sell threshold.
            roc_above_buy_threshold = trade_signals.at[index,
                                                       "roc"] > self.buy_threshold
            roc_below_sell_threshold = trade_signals.at[index,
                                                        "roc"] < self.sell_threshold

            # Determine buy and sell signals.
            buy_signal = roc_above_buy_threshold
            sell_signal = roc_below_sell_threshold

            # Record buy and sell signals.
            trade_signals.at[index, "buy_signal"] = buy_signal
            trade_signals.at[index, "sell_signal"] = sell_signal

        # Drop the ROC indicator.
        trade_signals = trade_signals.drop(columns=["roc"])

        return trade_signals


class Awesome_Oscillator_Bot:
    def __init__(self, ohlcv_df, window1, window2):
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
        # Initialises output column for the buy signals.
        trade_signals["buy_signal"] = False
        # Initialises output column for the sell signals.
        trade_signals["sell_signal"] = False

        for index, row in trade_signals.iterrows():
            # Evaluate literals
            # buying pressure
            awesome_above_zero = trade_signals.at[index, 'ao'] > 0
            # selling pressure
            awesome_below_zero = trade_signals.at[index, 'ao'] < 0

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

    def __init__(self, ohlcv_df, constituent_bot_parameters, number_of_conjunctions, strategies_used):
        super().__init__(ohlcv_df)
        self.constituent_bot_parameters = constituent_bot_parameters
        self.bot_type = "Ensemble"

        self.number_of_conjunctions = number_of_conjunctions
        self.strategies_used = []

        self.params = [number_of_conjunctions, strategies_used]

    def initialise_bots(self):
        all_bot_signals = {}

        for parameter_list in self.constituent_bot_parameters:
            # Get the bot name and remove it from the dictionary.
            parameter_list_copy = dict(parameter_list)
            bot_name = parameter_list_copy.pop('bot_name')
            # self.strategy_names.append(bot_name)
            self.strategies_used.append(bot_name)
            # Initialize the bot with its specified parameters and save output signals dataframe.
            signals_df = globals()[bot_name](
                self.ohlcv_df, **parameter_list_copy).generate_signals()
            all_bot_signals[bot_name] = signals_df

        return all_bot_signals

    def construct_conjunction(self, trade_type):
        # # Chooses the strategies used in the conjunction.
        # number_of_strategies_included = random.randint(self.min_literals, self.max_literals)
        # strategies_used = random.sample(self.strategy_names, number_of_strategies_included)

        # Constructs the conjunction by ANDing the signals from the selected strategies.
        buy_signals = []
        for strategy_name in self.strategies_used:
            bot_signals = f"all_bot_signals['{strategy_name}']"
            buy_signal = f"{bot_signals}.at[index, '{trade_type}_signal']"
            buy_signals.append(buy_signal)
        conjunction = " and ".join(buy_signals)

        return conjunction

    def construct_dnf(self, trade_type):
        # # Chooses how many conjunctions are used in the DNF.
        # number_of_conjunctions = random.randint(1, 4)

        # Constructs the DNF by generating conjunctions and ORing them together.
        conjunctions = []
        for i in range(self.number_of_conjunctions):
            conjunction = self.construct_conjunction(trade_type)
            conjunctions.append(conjunction)
        dnf = " or ".join(conjunctions)

        return dnf

    def generate_signals(self):
        # Creates a copy of the DataFrame to avoid modifying the original.
        trade_signals = self.ohlcv_df.copy()

        all_bot_signals = self.initialise_bots()

        # Create random DNF expression for buy signal.
        buy_dnf = self.construct_dnf(trade_type="buy")

        # Evaluate DNF expression for each day of data and save to dataframe.
        for index, row in trade_signals.iterrows():
            buy_dnf_with_index = buy_dnf.replace("index", str(index))
            buy_signal = eval(buy_dnf_with_index)
            trade_signals.at[index, "buy_signal"] = buy_signal

        # Create random DNF expression for sell signal.
        sell_dnf = self.construct_dnf(trade_type="sell")

        # Evaluate DNF expression for each day of data and save to dataframe.
        for index, row in trade_signals.iterrows():
            sell_dnf_with_index = sell_dnf.replace("index", str(index))
            sell_signal = eval(sell_dnf_with_index)
            trade_signals.at[index, "sell_signal"] = sell_signal

        # return buy_dnf, sell_dnf, trade_signals
        return trade_signals

    # def construct_conjunction(self, trade_type):
    #     # Chooses the strategies used in the conjunction.
    #     number_of_strategies_included = random.randint(self.min_literals, self.max_literals)
    #     strategies_used = random.sample(self.strategy_names, number_of_strategies_included)

    #     # Constructs the conjunction by ANDing the signals from the selected strategies.
    #     buy_signals = []
    #     for strategy_name in strategies_used:
    #         bot_signals = f"all_bot_signals['{strategy_name}']"
    #         buy_signal = f"{bot_signals}.at[index, '{trade_type}_signal']"
    #         buy_signals.append(buy_signal)
    #     conjunction = " and ".join(buy_signals)

    #     return conjunction

    # def construct_dnf(self, trade_type):
    #     # Chooses how many conjunctions are used in the DNF.
    #     number_of_conjunctions = random.randint(1, 4)

    #     # Constructs the DNF by generating conjunctions and ORing them together.
    #     conjunctions = []
    #     for i in range(number_of_conjunctions):
    #             conjunction = self.construct_conjunction(trade_type)
    #             conjunctions.append(conjunction)
    #     dnf = " or ".join(conjunctions)

    #     return dnf

    # def generate_signals(self):
    #     # Creates a copy of the DataFrame to avoid modifying the original.
    #     trade_signals = self.ohlcv_df.copy()

    #     all_bot_signals = self.initialise_bots()

    #     # Create random DNF expression for buy signal.
    #     buy_dnf = self.construct_dnf(trade_type = "buy")

    #     # Evaluate DNF expression for each day of data and save to dataframe.
    #     for index, row in trade_signals.iterrows():
    #         buy_dnf_with_index = buy_dnf.replace("index", str(index))
    #         buy_signal = eval(buy_dnf_with_index)
    #         trade_signals.at[index, "buy_signal"] = buy_signal

    #     # Create random DNF expression for sell signal.
    #     sell_dnf = self.construct_dnf(trade_type = "sell")

    #     # Evaluate DNF expression for each day of data and save to dataframe.
    #     for index, row in trade_signals.iterrows():
    #         sell_dnf_with_index = sell_dnf.replace("index", str(index))
    #         sell_signal = eval(sell_dnf_with_index)
    #         trade_signals.at[index, "sell_signal"] = sell_signal

    #     # return buy_dnf, sell_dnf, trade_signals
    #     return trade_signals


# class Simulate:
#     def __init__(self):
#         pass

#     def get_daily_ohlcv_data():
#         """ Fetches the most recent 720 days of OHLCV data on BTC/AUD from Kraken.
#             Converts data into a Pandas DataFrame with column titles.
#             Alters and returns the DataFrame for further analysis.
#         """
#         exchange = kraken()
#         ohlcv_data = exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
#         ohlcv_df = DataFrame(ohlcv_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
#         ohlcv_df["next_day_open"] = ohlcv_df["open"].shift(-1)     # Adds column for next day's open price.
#         ohlcv_df = ohlcv_df.iloc[:-1]    # Removes last day's data as the bot cannot trade the next day.

#         return ohlcv_df


#     def execute_trades(trade_signals, fee_percentage):
#         """ Executes all of the identified trade signals sequentially.
#             Ensures the final holdings are in AUD.
#             Returns the trading account's final balance in AUD.
#         """
#         trade_results = trade_signals.copy()
#         trade_results["portfolio_value"] = 0

#         aud_balance = 100.00
#         btc_balance = 0.00

#         last_trade = "sell"

#         # For each day:
#         for index, row in trade_results.iterrows():
#             buy_signal = row["buy_signal"]
#             sell_signal = row["sell_signal"]
#             next_day_open_price = row["next_day_open"]

#             # Records daily portfolio value in AUD at market close.
#             if last_trade == "buy":
#                 trade_results.at[index, "portfolio_value"] = btc_balance * row["close"]
#             elif last_trade == "sell":
#                 trade_results.at[index, "portfolio_value"] = aud_balance

#             # Executes trade at following day's open price if today's data results in trade signal.
#             if buy_signal == True and last_trade == "sell":  # Buy signal
#                 # Converts all AUD to BTC using the next day's open price and applies percentage fee.
#                 btc_balance = aud_balance / next_day_open_price * (1 - fee_percentage)
#                 aud_balance = 0
#                 last_trade = "buy"

#             elif sell_signal == True and last_trade == "buy":  # Sell signal
#                 # Converts all BTC to AUD using the next day's open price and applies percentage fee.
#                 aud_balance = btc_balance * next_day_open_price * (1 - fee_percentage)
#                 btc_balance = 0
#                 last_trade = "sell"

#         # Converts final holdings to AUD using final day's open price if final holdings are in BTC.
#         if last_trade == "buy":
#             last_close_price = trade_results["next_day_open"].iloc[-1]
#             aud_balance = btc_balance * last_close_price * (1 - fee_percentage)
#             btc_balance = 0

#         return aud_balance, trade_results


#     def plot_trading_simulation(trade_results):
#         # Create a figure and axis
#         fig, ax = plt.subplots()

#         # Set the x-axis data (day of trading) and y-axis data (portfolio value in AUD at close)
#         x_data = trade_results.index
#         y_data = trade_results["portfolio_value"]

#         # Plot the data
#         ax.plot(x_data, y_data)

#         # Set the labels and title
#         ax.set_xlabel("Day of Trading")
#         ax.set_ylabel("Portfolio Value in AUD at Close")
#         ax.set_title("Trading Simulation Results")

#         # Display the plot
#         plt.show()


# class Bots:
#     def __init__(self, ohlcv_df):
#         self.ohlcv_df = ohlcv_df

#     def MACD_bot(self, slow_window, fast_window, signal_window):
#         """ Computes the MACD histogram using the daily close prices.
#             Identifies the buy/sell signals (changes in histogram sign).
#             Returns a DataFrame with all the required data for executing the trades.
#         """
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The MACD histogram is computed from the daily close prices.
#         close_prices = trade_signals["close"]

#         # Computes MACD histogram.
#         macd_indicator = MACD(close_prices, window_slow = slow_window, window_fast = fast_window, window_sign = signal_window)
#         trade_signals["MACD_histogram"] = macd_indicator.macd_diff()    # Computes indicator values.

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iloc[1:].iterrows():
#             # Evaluates literals.
#             MACD_histogram_was_negative = 0 > trade_signals.at[index - 1, "MACD_histogram"]
#             MACD_histpgram_was_positive = trade_signals.at[index - 1, "MACD_histogram"] > 0
#             MACD_histogram_now_negative = 0 > trade_signals.at[index, "MACD_histogram"]
#             MACD_histogram_now_positive = trade_signals.at[index, "MACD_histogram"] > 0

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = MACD_histogram_was_negative and MACD_histogram_now_positive
#             sell_signal = MACD_histpgram_was_positive and MACD_histogram_now_negative

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted column from output dataframe.
#         trade_signals = trade_signals.drop(columns = ["MACD_histogram"])

#         return trade_signals


#     def bollinger_bands_bot(self, window, num_standard_deviations):
#         """ Computes the Bollinger band values using the daily close prices.
#             Identifies the buy/sell signals (price exiting the bands).
#             Returns a DataFrame with all the required data for executing the trades.
#         """
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The Bollinger Bands are computed from the daily close prices.
#         close_prices = trade_signals["close"]

#         # Computes Bollinger Bands indicators.
#         bb_indicator = BollingerBands(close_prices, window = window, window_dev = num_standard_deviations)
#         trade_signals["BB_highband"] = bb_indicator.bollinger_hband()
#         trade_signals["BB_lowband"] = bb_indicator.bollinger_lband()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iterrows():
#             # Evaluates literals.
#             price_above_BB_highband = trade_signals.at[index, "close"] > trade_signals.at[index, "BB_highband"]
#             price_below_BB_lowband = trade_signals.at[index, "BB_lowband"] > trade_signals.at[index, "close"]

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = price_below_BB_lowband
#             sell_signal = price_above_BB_highband

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals
#         trade_signals = trade_signals.drop(columns=["BB_highband", "BB_lowband"])

#         return trade_signals


#     def RSI_bot(self, overbought_threshold, oversold_threshold, window):
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The RSI values are computed from the daily close prices.
#         close_prices = trade_signals["close"]

#         # Computes RSI values.
#         rsi_indicator = RSIIndicator(close_prices, window = window)
#         trade_signals['rsi'] = rsi_indicator.rsi()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iterrows():
#             # Evaluates literals.
#             rsi_above_overbought_threshold = trade_signals.at[index, 'rsi'] > overbought_threshold
#             rsi_below_oversold_threshold = oversold_threshold > trade_signals.at[index, 'rsi']

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = rsi_below_oversold_threshold
#             sell_signal = rsi_above_overbought_threshold

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals.
#         trade_signals = trade_signals.drop(columns=['rsi'])

#         return trade_signals


#     def VWAP_bot(self, window):
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The VWAP values are computed from the high, low, close and volume data.
#         vwap_indicator = VolumeWeightedAveragePrice(
#             high = trade_signals['high'],
#             low = trade_signals['low'],
#             close = trade_signals['close'],
#             volume = trade_signals['volume'],
#             window = window
#         )
#         trade_signals['vwap'] = vwap_indicator.volume_weighted_average_price()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iterrows():
#             # Evaluates literals.
#             price_above_vwap = trade_signals.at[index, 'close'] > trade_signals.at[index, 'vwap']
#             price_below_vwap = trade_signals.at[index, 'vwap'] > trade_signals.at[index, 'close']

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = price_below_vwap
#             sell_signal = price_above_vwap

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals.
#         trade_signals = trade_signals.drop(columns=['vwap'])

#         return trade_signals


#     def stochastic_oscillator_bot(self, oscillator_window, signal_window, overbought_threshold, oversold_threshold):
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The stochastic oscillator values are computed from the high, low, and close prices.
#         StochOsc = StochasticOscillator(
#             trade_signals['close'],
#             trade_signals['high'],
#             trade_signals['low'],
#             window = oscillator_window,
#             smooth_window = signal_window
#         )
#         trade_signals["stoch_oscillator"] = StochOsc.stoch()
#         trade_signals["stoch_signal"] = StochOsc.stoch_signal()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iloc[1:].iterrows():
#             # Evaluates literals.
#             stoch_oscillator_oversold = oversold_threshold > trade_signals.at[index, "stoch_oscillator"]
#             stoch_signal_oversold = oversold_threshold > trade_signals.at[index, "stoch_signal"]
#             oscillator_was_above_signal = trade_signals.at[index - 1, "stoch_oscillator"] > trade_signals.at[index - 1, "stoch_signal"]
#             oscillator_now_below_signal = trade_signals.at[index, "stoch_signal"] > trade_signals.at[index, "stoch_oscillator"]

#             stoch_oscillator_overbought = trade_signals.at[index, "stoch_oscillator"] > overbought_threshold
#             stoch_signal_overbought = trade_signals.at[index, "stoch_signal"] > overbought_threshold
#             oscillator_was_below_signal = trade_signals.at[index - 1, "stoch_signal"] > trade_signals.at[index - 1, "stoch_oscillator"]
#             oscillator_now_above_signal = trade_signals.at[index, "stoch_oscillator"] > trade_signals.at[index, "stoch_signal"]

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = stoch_oscillator_oversold and stoch_signal_oversold and oscillator_was_above_signal and oscillator_now_below_signal
#             sell_signal = stoch_oscillator_overbought and stoch_signal_overbought and oscillator_was_below_signal and oscillator_now_above_signal

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals.
#         trade_signals = trade_signals.drop(columns=["stoch_oscillator", "stoch_signal"])

#         return trade_signals


# class Ensemble:
#     def __init__(self, ohlcv_df, *args):
#         self.ohlcv_df = ohlcv_df
#         self.conjunction_parameters = args

#     def determine_bot_signals(self, all_parameters):
#         all_bot_signals = {}
#         strategy_names = []

#         for parameter_list in all_parameters:
#             # Get the bot name and remove it from the dictionary.
#             bot_name = parameter_list.pop('bot_name')
#             strategy_names.append(bot_name)
#             # Run the bot with its specified parameters and save output signals dataframe.
#             signals_df = getattr(Bots(self.ohlcv_df), bot_name)(**parameter_list)
#             all_bot_signals[bot_name] = signals_df

#         return all_bot_signals, strategy_names


#     def construct_conjunction(self, strategy_names, trade_type, min_literals, max_literals):
#         # Chooses the strategies used in the conjunction.
#         number_of_strategies_included = random.randint(min_literals, max_literals)
#         strategies_used = random.sample(strategy_names, number_of_strategies_included)

#         # Constructs the conjunction by ANDing the signals from the selected strategies.
#         buy_signals = []
#         for strategy_name in strategies_used:
#             buy_signal = f"{strategy_name}.at[index, '{trade_type}_signal']"
#             buy_signals.append(buy_signal)
#         conjunction = " and ".join(buy_signals)

#         return conjunction

##########################################################################################################################

# class Bots:
#     def __init__(self, ohlcv_df):
#         self.ohlcv_df = ohlcv_df

#     def MACD_bot(self, slow_window, fast_window, signal_window):
#         """ Computes the MACD histogram using the daily close prices.
#             Identifies the buy/sell signals (changes in histogram sign).
#             Returns a DataFrame with all the required data for executing the trades.
#         """
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The MACD histogram is computed from the daily close prices.
#         close_prices = trade_signals["close"]

#         # Computes MACD histogram.
#         macd_indicator = MACD(close_prices, window_slow = slow_window, window_fast = fast_window, window_sign = signal_window)
#         trade_signals["MACD_histogram"] = macd_indicator.macd_diff()    # Computes indicator values.

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iloc[1:].iterrows():
#             # Evaluates literals.
#             MACD_histogram_was_negative = 0 > trade_signals.at[index - 1, "MACD_histogram"]
#             MACD_histpgram_was_positive = trade_signals.at[index - 1, "MACD_histogram"] > 0
#             MACD_histogram_now_negative = 0 > trade_signals.at[index, "MACD_histogram"]
#             MACD_histogram_now_positive = trade_signals.at[index, "MACD_histogram"] > 0

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = MACD_histogram_was_negative and MACD_histogram_now_positive
#             sell_signal = MACD_histpgram_was_positive and MACD_histogram_now_negative

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted column from output dataframe.
#         trade_signals = trade_signals.drop(columns = ["MACD_histogram"])

#         return trade_signals


#     def bollinger_bands_bot(self, window, num_standard_deviations):
#         """ Computes the Bollinger band values using the daily close prices.
#             Identifies the buy/sell signals (price exiting the bands).
#             Returns a DataFrame with all the required data for executing the trades.
#         """
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The Bollinger Bands are computed from the daily close prices.
#         close_prices = trade_signals["close"]

#         # Computes Bollinger Bands indicators.
#         bb_indicator = BollingerBands(close_prices, window = window, window_dev = num_standard_deviations)
#         trade_signals["BB_highband"] = bb_indicator.bollinger_hband()
#         trade_signals["BB_lowband"] = bb_indicator.bollinger_lband()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iterrows():
#             # Evaluates literals.
#             price_above_BB_highband = trade_signals.at[index, "close"] > trade_signals.at[index, "BB_highband"]
#             price_below_BB_lowband = trade_signals.at[index, "BB_lowband"] > trade_signals.at[index, "close"]

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = price_below_BB_lowband
#             sell_signal = price_above_BB_highband

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals
#         trade_signals = trade_signals.drop(columns=["BB_highband", "BB_lowband"])

#         return trade_signals


#     def RSI_bot(self, overbought_threshold, oversold_threshold, window):
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The RSI values are computed from the daily close prices.
#         close_prices = trade_signals["close"]

#         # Computes RSI values.
#         rsi_indicator = RSIIndicator(close_prices, window = window)
#         trade_signals['rsi'] = rsi_indicator.rsi()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iterrows():
#             # Evaluates literals.
#             rsi_above_overbought_threshold = trade_signals.at[index, 'rsi'] > overbought_threshold
#             rsi_below_oversold_threshold = oversold_threshold > trade_signals.at[index, 'rsi']

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = rsi_below_oversold_threshold
#             sell_signal = rsi_above_overbought_threshold

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals.
#         trade_signals = trade_signals.drop(columns=['rsi'])

#         return trade_signals


#     def VWAP_bot(self, window):
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The VWAP values are computed from the high, low, close and volume data.
#         vwap_indicator = VolumeWeightedAveragePrice(
#             high = trade_signals['high'],
#             low = trade_signals['low'],
#             close = trade_signals['close'],
#             volume = trade_signals['volume'],
#             window = window
#         )
#         trade_signals['vwap'] = vwap_indicator.volume_weighted_average_price()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iterrows():
#             # Evaluates literals.
#             price_above_vwap = trade_signals.at[index, 'close'] > trade_signals.at[index, 'vwap']
#             price_below_vwap = trade_signals.at[index, 'vwap'] > trade_signals.at[index, 'close']

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = price_below_vwap
#             sell_signal = price_above_vwap

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals.
#         trade_signals = trade_signals.drop(columns=['vwap'])

#         return trade_signals


#     def stochastic_oscillator_bot(self, oscillator_window, signal_window, overbought_threshold, oversold_threshold):
#         # Creates a copy of the DataFrame to avoid modifying the original.
#         trade_signals = self.ohlcv_df.copy()

#         # The stochastic oscillator values are computed from the high, low, and close prices.
#         StochOsc = StochasticOscillator(
#             trade_signals['close'],
#             trade_signals['high'],
#             trade_signals['low'],
#             window = oscillator_window,
#             smooth_window = signal_window
#         )
#         trade_signals["stoch_oscillator"] = StochOsc.stoch()
#         trade_signals["stoch_signal"] = StochOsc.stoch_signal()

#         # Initialises output columns.
#         trade_signals["buy_signal"] = False    # Initialises output column for the buy signals.
#         trade_signals["sell_signal"] = False     # Initialises output column for the sell signals.

#         for index, row in trade_signals.iloc[1:].iterrows():
#             # Evaluates literals.
#             stoch_oscillator_oversold = oversold_threshold > trade_signals.at[index, "stoch_oscillator"]
#             stoch_signal_oversold = oversold_threshold > trade_signals.at[index, "stoch_signal"]
#             oscillator_was_above_signal = trade_signals.at[index - 1, "stoch_oscillator"] > trade_signals.at[index - 1, "stoch_signal"]
#             oscillator_now_below_signal = trade_signals.at[index, "stoch_signal"] > trade_signals.at[index, "stoch_oscillator"]

#             stoch_oscillator_overbought = trade_signals.at[index, "stoch_oscillator"] > overbought_threshold
#             stoch_signal_overbought = trade_signals.at[index, "stoch_signal"] > overbought_threshold
#             oscillator_was_below_signal = trade_signals.at[index - 1, "stoch_signal"] > trade_signals.at[index - 1, "stoch_oscillator"]
#             oscillator_now_above_signal = trade_signals.at[index, "stoch_oscillator"] > trade_signals.at[index, "stoch_signal"]

#             # Evaluates buy and sell conjunctions to determine buy and sell signals.
#             buy_signal = stoch_oscillator_oversold and stoch_signal_oversold and oscillator_was_above_signal and oscillator_now_below_signal
#             sell_signal = stoch_oscillator_overbought and stoch_signal_overbought and oscillator_was_below_signal and oscillator_now_above_signal

#             # Records buy and sell signals.
#             trade_signals.at[index, "buy_signal"] = buy_signal
#             trade_signals.at[index, "sell_signal"] = sell_signal

#         # Drops the unwanted columns from trade_signals.
#         trade_signals = trade_signals.drop(columns=["stoch_oscillator", "stoch_signal"])

#         return trade_signals


# class Ensemble:
#     def __init__(self, ohlcv_df):
#         self.ohlcv_df = ohlcv_df

#     def determine_bot_signals(self, all_parameters):
#         all_bot_signals = {}
#         strategy_names = []

#         for parameter_list in all_parameters:
#             # Get the bot name and remove it from the dictionary.
#             bot_name = parameter_list.pop('bot_name')
#             strategy_names.append(bot_name)
#             # Run the bot with its specified parameters and save output signals dataframe.
#             signals_df = getattr(Bots(self.ohlcv_df), bot_name)(**parameter_list)
#             all_bot_signals[bot_name] = signals_df

#         return all_bot_signals, strategy_names


#     def construct_conjunction(self, strategy_names, trade_type, min_literals, max_literals):
#         # Chooses the strategies used in the conjunction.
#         number_of_strategies_included = random.randint(min_literals, max_literals)
#         strategies_used = random.sample(strategy_names, number_of_strategies_included)

#         # Constructs the conjunction by ANDing the signals from the selected strategies.
#         buy_signals = []
#         for strategy_name in strategies_used:
#             buy_signal = f"{strategy_name}.at[index, '{trade_type}_signal']"
#             buy_signals.append(buy_signal)
#         conjunction = " and ".join(buy_signals)

#         return conjunction
