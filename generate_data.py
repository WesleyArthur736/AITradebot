import ccxt
import pandas as pd

kraken_exchange = ccxt.kraken()

btc_aud_ohlcv = kraken_exchange.fetch_ohlcv("BTC/AUD", timeframe="1d")
btc_aud_df = pd.DataFrame(btc_aud_ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
btc_aud_df.to_csv("data/btc_aud_1d.csv", index=False)
