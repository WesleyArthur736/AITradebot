from macd_trader import MACDTrader
from simulate import Simulate

macd_trader_1 = MACDTrader(window_slow=26, window_fast=12)
macd_trader_2 = MACDTrader(window_slow=30, window_fast=10)

simulation_1 = Simulate(trader=macd_trader_1)
simulation_1.run_simulation()

simulation_2 = Simulate(trader=macd_trader_2)
simulation_2.run_simulation()

print("Simulation 1 final position: ", simulation_1.net_worth)
print("Simulation 2 final position: ", simulation_2.net_worth)
