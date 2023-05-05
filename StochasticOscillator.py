import ccxt
from ta.momentum import StochasticOscillator
import pandas as pd
import numpy as np
import copy

def main():
    STOCH_OSCILL_WINDOW = 14
    STOCH__OSCILL_WINDOW =3
    MONEY = 100
    
    bitcoin_data, sosc_data, so_signal = set_up_objects([STOCH_OSCILL_WINDOW, STOCH__OSCILL_WINDOW])
    soscill_bot = stochasticOscillator(bitcoin_data, sosc_data, so_signal, MONEY)
    
    sell_triggers, buy_triggers = soscill_bot.get_triggers()
    orders = soscill_bot.formulate_orders(sell_triggers, buy_triggers)
    soscill_bot.execute(orders)
    '''
    stochOsc = StochasticOscillator(bitcoin_indicators['close'],bitcoin_indicators['high'],bitcoin_indicators['low'])
    sosc_data = stochOsc.stoch()
    so_signal = stochOsc.stoch_signal()
    '''
    
    #act_crossover = so_crossover_sig[so_crossover_sig==act_crossover]
    

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
        
    def get_triggers(self):
        close_price = self.bitcoin_Data['close']
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
        orders = np.zeros((720))
        for tup in sell_triggers:
            print("tup",tup[0])
            
            orders[tup[0]]=tup[1]
                
        for tup in sell_triggers:
            
            orders[tup[0]]=tup[1]
        
        return pd.Series(orders.astype(int), name ="trade_signal")
    
    def execute(self, orders):
        current_money=[]
        ndot = self.bitcoin_Data['open'][1:719]
        print(ndot)
        return
    '''
        if sell_trigger_status == True and self.coins > 0:
                potential_funds= current_price.iloc[day] * self.coins
                funds_gained = potential_funds - (potential_funds/50)
                self.cash = float(self.cash + funds_gained)
                self.coins = 0
                
                
            elif buy_trigger_status == True:
                available_funds = self.cash - (self.cash/50)
                self.coins = float(self.coins + (available_funds/ current_price.iloc[day]))
                self.cash = 0 #float((self.cash - (self.cash//current_price[day]) * current_price[day]))
                
            print("prev trans", previous_transaction_day,"day", day)
            
        if self.coins > 0:
            self.cash = float(self.cash + (current_price.iloc[day] * self.coins))
            self.coins = 0
            '''
        
main()