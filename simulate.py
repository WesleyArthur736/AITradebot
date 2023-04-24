import pandas as pd
import matplotlib.pyplot as plt

from macd_trader import MACDTrader
from models import Trader

# 720 days of data
# start date: Wednesday, 5 May 2021
# end date: Monday, 24 April 2023
btc_aud_df = pd.read_csv("data/btc_aud_1d.csv")
btc_aud_df["Mid Price"] = 0.5*btc_aud_df["High"] + 0.5*btc_aud_df["Close"]


def simulate(trader: Trader, start=0, end=500):
    pnls = []
    position = {"BTC": 0, "AUD": 100.0}
    trading_data = btc_aud_df.loc[start:(end - 1)]
    signals = trader.generate_signals(trading_data)

    order = 0
    for idx, row in trading_data.iterrows():
        if order == 1:
            price = row["Open"]
            position["BTC"] = position["AUD"] / price
            position["AUD"] = 0
            order = 0
        elif order == -1:
            price = row["Open"]
            position["AUD"] = position["BTC"] * price
            position["BTC"] = 0
            order = 0
        signal = signals[idx]
        # buy
        if signal == 1.0:
            print("BUY AT ", idx)
            order = 1
        # sell
        elif signal == -1.0:
            print("SELL AT ", idx)
            order = -1

        pnls.append(position["BTC"] * row["Close"] + position["AUD"])

    plt.plot(pnls)
    plt.show()

    final_close = trading_data.iloc[-1]["Close"]
    print("Final PNL: ", position["BTC"]*final_close + position["AUD"])


if __name__ == "__main__":
    macd_trader = MACDTrader()
    simulate(macd_trader)
