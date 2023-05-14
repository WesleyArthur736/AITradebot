# AITradebot
This Project Develops an AI bot optimised to trade BTCAUD

The main script is in 'main_script.py'. In the main section of 'main_script.py' there are several functions that one can uncomment to run either an individual bot or an ensemble bot.
running 'run_macd_optimized'will run one instance of an un-optimized MACD bot and one instance of a GA-optimized MACD bot.

Running 'run_ensemble_bots_non_optimal_and_optimal' will run one instance of the ensemble bot on randomly initialised constituent bots, and one instance of the GA algorithm for optimising the ensemble bot's DNF statements.
