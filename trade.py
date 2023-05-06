import ccxt
from ta.momentum import RSIIndicator
import pandas as pd


def get_data():
    exchange = ccxt.kraken()
    bars = exchange.fetch_ohlcv('BTC/AUD', timeframe="1d", limit=720)
    df = pd.DataFrame(
        bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['next_day_open'] = df['open'].shift(-1)
    df = df.iloc[:-1]
    return df


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
            trade_results.at[index,
                             "portfolio_value"] = btc_balance * row["close"]
        else:
            trade_results.at[index, "portfolio_value"] = aud_balance

        # Executes trade at following day's open price if today's data results in trade signal.
        if trading_signal == 1 and last_trade == -1:  # Buy signal
            # Converts all AUD to BTC using the next day's open price and applies percentage fee.
            btc_balance = aud_balance / \
                next_day_open_price * (1 - fee_percentage)
            aud_balance = 0
            last_trade = 1

        elif trading_signal == -1 and last_trade == 1:  # Sell signal
            # Converts all BTC to AUD using the next day's open price and applies percentage fee.
            aud_balance = btc_balance * \
                next_day_open_price * (1 - fee_percentage)
            btc_balance = 0
            last_trade = -1

    # Converts final holdings to AUD using final day's open price if final holdings are in BTC.
    if aud_balance == 0:
        last_close_price = trade_results["next_day_open"].iloc[-1]
        aud_balance = btc_balance * last_close_price * (1 - fee_percentage)
        btc_balance = 0

    return aud_balance, trade_results
