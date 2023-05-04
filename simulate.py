import pandas as pd
import matplotlib.pyplot as plt

from macd_trader import MACDTrader
from vwap_trader import VWAPTrader
from models import Trader


class Simulate:
    def __init__(self, trader: Trader, start=0, end=500, fee=0.02):
        # 720 days of data
        # start date: Wednesday, 5 May 2021
        # end date: Monday, 24 April 2023
        btc_aud_df = pd.read_csv("data/btc_aud_1d.csv")
        btc_aud_df["Mid Price"] = 0.5 * btc_aud_df["High"] + 0.5 * btc_aud_df["Close"]
        trading_data = btc_aud_df.iloc[start:(end - 1)].reset_index(drop=False)
        self.trading_data = trading_data
        self.trader = trader
        self.fee = fee

        self.position = {"BTC": 0, "AUD": 100.0}
        self.net_worth = 100.0
        self.pnls = []

    def run_simulation(self):
        signals = self.trader.generate_signals(self.trading_data)

        order = 0
        for idx, row in self.trading_data.iterrows():
            if order == 1:
                price = row["Open"]
                self.position["BTC"] = self.position["AUD"] / price * (1 - self.fee)
                self.position["AUD"] = 0
                order = 0
            elif order == -1:
                price = row["Open"]
                self.position["AUD"] = self.position["BTC"] * price * (1 - self.fee)
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
    macd_trader = MACDTrader(window_slow=26, window_fast=12)
    vwap_trader = VWAPTrader(vwap_window=14)
    simulation = Simulate(trader=vwap_trader, start=0, end=500, fee=0)
    simulation.run_simulation()
    plt.plot(simulation.pnls)
    plt.show()
    print(simulation.net_worth)
