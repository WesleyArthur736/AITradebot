import ccxt
from ta.momentum import StochasticOscillator
import pandas as pd
import numpy as np
import copy

def main():
    STOCH_OSCILL_WINDOW = 14
    STOCH__OSCILL_WINDOW = 3
    MONEY = 100
    
    bitcoin_data, sosc_data, so_signal = set_up_objects([STOCH_OSCILL_WINDOW, STOCH__OSCILL_WINDOW])
    soscill_bot = stochasticOscillator(bitcoin_data, sosc_data, so_signal, MONEY)
    
    sell_triggers, buy_triggers = soscill_bot.get_triggers()
    orders = soscill_bot.formulate_orders(sell_triggers, buy_triggers)
    portfolio_hist, ndo = soscill_bot.execute(orders)
    bitcoin_data= bitcoin_data[0:719]
    
    orders.reset_index(drop=True,inplace=True)
    portfolio_hist.reset_index(drop=True,inplace=True)
    ndo.reset_index(drop=True,inplace=True)
    
    bot_data = pd.concat([bitcoin_data, ndo, orders, portfolio_hist], axis = 1)
    print(bot_data)
    
    return bot_data
    '''
    stochOsc = StochasticOscillator(bitcoin_indicators['close'],bitcoin_indicators['high'],bitcoin_indicators['low'])
    sosc_data = stochOsc.stoch()
    so_signal = stochOsc.stoch_signal()
    '''
    
    
    

def set_up_objects(lst):
    kraken_exchange = ccxt.kraken()
    kraken_market = kraken_exchange.load_markets()
    
    bitcoin_data = kraken_exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
    bitcoin_indicators = pd.DataFrame(bitcoin_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
    
    stochOsc = StochasticOscillator(bitcoin_indicators['close'],bitcoin_indicators['high'],bitcoin_indicators['low'],window=lst[0],smooth_window=lst[1])
    sosc_data = stochOsc.stoch()
    so_signal = stochOsc.stoch_signal()
    
    return bitcoin_indicators, sosc_data, so_signal

class stochasticOscillator:
    def __init__(self,bitcoin_data, sosc_data, so_signal, MONEY ):
        self.bitcoin_Data =bitcoin_data
        self.stochOsc= sosc_data
        self.so_signal= so_signal
        self.cash= MONEY
        self.coins = 0
        
    def get_triggers(self):
        
        sell_triggers = []
        buy_triggers = []
        sosc= self.stochOsc.to_numpy()
        so_s= self.so_signal.to_numpy()
        
        # prepare arrays to give point in ranges of significance to the Stochastic Oscillator index
        so_overbought = np.argwhere(sosc>80)
        so_oversold = np.argwhere(sosc<20)
        
        so_crossover_sig = np.argwhere(np.greater(so_s , sosc))
        crossover_back = np.argwhere(np.greater(sosc , so_s))
        crossover_booleans_osc = np.in1d(crossover_back, so_crossover_sig + 1 )
        crossover_booleans_sig= np.in1d(so_crossover_sig,crossover_back +1)
        
        # Determine the triggers for selling
        # Output a list of selling triggers
        for inx in so_overbought: 
            if np.any(crossover_back == inx) == True: 
                idx_crossover = np.where(crossover_back == inx) #suspect contradiction
                print(idx_crossover[0], inx)
                
                if crossover_booleans_osc[idx_crossover[0]] == True:
                    print("i made it to here")
                    if so_s[inx] < sosc[inx]: 
                        #print("loading variable")
                        sell_triggers.append((inx[0], "-1"))
        print(sell_triggers)
        
        # Determine the triggers for buying
        for inx in so_oversold: 
            if np.any(so_crossover_sig == inx) == True:
                idx_crossover = np.where(so_crossover_sig == inx)
                
                if crossover_booleans_sig[idx_crossover[0]] == True:
                    if so_s[inx] > sosc[inx]:
                        buy_triggers.append((inx[0], 1))
                    
        print (buy_triggers)      
        
        return sell_triggers,buy_triggers 
    
        
    
    def formulate_orders(self, sell_triggers,buy_triggers):
        NO_DAYS_TRADE = 719
        orders = np.zeros((NO_DAYS_TRADE))
        for tup in buy_triggers:
            if tup[0] < NO_DAYS_TRADE:
                
                print("tup",tup[0])
            
                orders[tup[0]]=tup[1]
                
        for tup in sell_triggers:
            if tup[0] < NO_DAYS_TRADE:
                orders[tup[0]]=tup[1]
        
        return pd.Series(orders.astype(int), name ="trade_signal")
    
    def execute(self, orders):
        portfolio_value=[]
        ndo = self.bitcoin_Data['open'][1:720]
        print(ndo)
        for ord in orders:
            print("orders", ord)
        
        # but Trigger
        for commd_idx in range(len(orders)):
            
            #buy_trigger
            if orders[commd_idx] == 1 and self.cash > 0:
                available_funds = self.cash - (self.cash/50)
                self.coins = float(self.coins + (available_funds/ ndo.iloc[commd_idx]))
                self.cash = 0 
            # sell trigger  
            elif orders[commd_idx] == -1 and self.coins > 0:
                potential_funds= ndo.iloc[commd_idx] * self.coins
                funds_gained = potential_funds - (potential_funds/50)
                self.cash = float(self.cash + funds_gained)
                self.coins = 0
            portfolio_value.append(self.cash)
                
        if self.coins > 0:
            self.cash = float(self.cash + (ndo.iloc[commd_idx] * self.coins))
            self.coins = 0   
            portfolio_value[len(portfolio_value)-1] = self.cash
        print(self.cash,self.coins)
        
        
        
        return pd.Series(portfolio_value, name="portfolio_value"), pd.Series(ndo, name = "next_day_open")
    
        
main()