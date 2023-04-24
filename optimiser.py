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



max_net_worth = 0
best_i = 0
best_j = 0
for i in range(20,50):
    for j in range(5,15):
        macd_trader = MACDTrader(window_slow=i, window_fast=j)
        simulation = Simulate(trader=macd_trader)
        simulation.run_simulation()
        if simulation.net_worth > max_net_worth:
            max_net_worth = simulation.net_worth
            best_i = i
            best_j = j

print(max_net_worth)
print(best_i)
print(best_j)
