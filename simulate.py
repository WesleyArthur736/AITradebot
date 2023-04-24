import pandas as pd
import matplotlib.pyplot as plt

from macd_trader import MACDTrader
from models import Trader


class Simulate:
    def __init__(self, trader: Trader, start=0, end=500):
        # 720 days of data
        # start date: Wednesday, 5 May 2021
        # end date: Monday, 24 April 2023
        btc_aud_df = pd.read_csv("data/btc_aud_1d.csv")
        btc_aud_df["Mid Price"] = 0.5 * btc_aud_df["High"] + 0.5 * btc_aud_df["Close"]
        trading_data = btc_aud_df.loc[start:(end - 1)].reset_index(drop=False)
        self.trading_data = trading_data
        self.trader = trader

        self.position = {"BTC": 0, "AUD": 100.0}
        self.net_worth = 100.0
        self.pnls = []

    def run_simulation(self):
        signals = self.trader.generate_signals(self.trading_data)

        order = 0
        for idx, row in self.trading_data.iterrows():
            if order == 1:
                price = row["Open"]
                self.position["BTC"] = self.position["AUD"] / price
                self.position["AUD"] = 0
                order = 0
            elif order == -1:
                price = row["Open"]
                self.position["AUD"] = self.position["BTC"] * price
                self.position["BTC"] = 0
                order = 0
            signal = signals[idx]
            # buy
            if signal == 1.0:
                # print("BUY AT ", idx)
                order = 1
            # sell
            elif signal == -1.0:
                # print("SELL AT ", idx)
                order = -1

            self.net_worth = self.position["BTC"] * row["Close"] + self.position["AUD"]
            self.pnls.append(self.net_worth)


if __name__ == "__main__":
    macd_trader = MACDTrader(window_slow=44, window_fast=7)
    simulation = Simulate(trader=macd_trader, start=501, end=600)
    simulation.run_simulation()
    plt.plot(simulation.pnls)
    plt.show()
    print(simulation.net_worth)
