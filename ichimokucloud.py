import ccxt
from ta.trend import IchimokuIndicator
import pandas as pd
import numpy as np
import copy

def main():
    ICHIMOKU_WINDOW1 = 9
    ICHIMOKU_WINDOW2 = 26
    ICHIMOKU_WINDOW3 = 52
    MONEY = 100
    
    soscill_bot = IchimokuInd(ICHIMOKU_WINDOW1, ICHIMOKU_WINDOW2, ICHIMOKU_WINDOW3)
    #soscill_bot = stochasticOscillator(bitcoin_data, sosc_data, so_signal, MONEY)
    bitcoin_data, spanA,spanB, base_line, conversion_line = soscill_bot.get_data()
    
    spanA = pd.Series(spanA, name="spanA")
    spanB = pd.Series(spanB, name="spanB")
    base_line = pd.Series(base_line, name="Baseline")
    conversion_line = pd.Series(conversion_line, name="ConversionLine")
    bitcoin_data = pd.concat([bitcoin_data,spanA,spanB, base_line, conversion_line],axis=1)
    #sell_triggers, buy_triggers 
    #orders = soscill_bot.formulate_orders(sell_triggers, buy_triggers)
    ndo, sell_signal, buy_signal = soscill_bot.execute_trades(bitcoin_data, 0.02)
    bitcoin_data= bitcoin_data[0:719]
    
    buy_signal.reset_index(drop=True,inplace=True)
    sell_signal.reset_index(drop=True,inplace=True)
    ndo.reset_index(drop=True,inplace=True)
    
    bot_data = pd.concat([bitcoin_data, ndo, buy_signal, sell_signal], axis = 1)
    print(bot_data)
    
    return bot_data
    
    
    
    
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
'''

class IchimokuInd:
    def __init__(self,ICHIMOKU_WINDOW1, ICHIMOKU_WINDOW2, ICHIMOKU_WINDOW3):
        STARTING_CASH = 100
        self.wind1 = ICHIMOKU_WINDOW1
        self.wind2 = ICHIMOKU_WINDOW2
        self.wind3 = ICHIMOKU_WINDOW3
        
        self.cash= STARTING_CASH
        self.coins = 0
        '''
        self.bitcoin_Data =bitcoin_data
        self.stochOsc= sosc_data
        self.so_signal= so_signal
        self.stochOsc=None
        self.so_signal=None
        
        '''
        
    def get_data(self):
        kraken_exchange = ccxt.kraken()
        kraken_market = kraken_exchange.load_markets()
        
        bitcoin_data = kraken_exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
        bitcoin_indicators = pd.DataFrame(bitcoin_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
        
        Ichimoku = IchimokuIndicator(bitcoin_indicators['high'],bitcoin_indicators['low'],window1=self.wind1,window2=self.wind2, window3=self.wind3)
        spanA= Ichimoku.ichimoku_a()
        spanB= Ichimoku.ichimoku_b()
        base_line= Ichimoku.ichimoku_base_line()
        conversion_line= Ichimoku.ichimoku_conversion_line()
        
        return bitcoin_indicators, spanA,  spanB, base_line,conversion_line
        
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
    
    def execute_trades(self, trade_signals, fee_percentage):
        portfolio_value=[]
        sell_signal = []
        buy_signal = []
        spanA= trade_signals["spanA"].to_numpy()
        spanB= trade_signals["spanB"].to_numpy()
        baseline= trade_signals["Baseline"].to_numpy()
        conversionline = trade_signals["ConversionLine"].to_numpy()
        ndo = trade_signals['open'][1:720]
        print(ndo)
        
        
        # but Trigger
        for day in range(len(ndo)):
            crossover=[spanA[day] > spanB[day], spanA[day+1] > spanB[day+1]]

            buy_trigger =(crossover[0] == False and crossover[1] == True ) and ndo[day]<spanA[day] and ndo[day]< spanB[day] and baseline[day] > conversionline[day]
            sell_trigger = (crossover[0] == True and crossover[1] == False) and ndo[day] >spanA[day] and ndo[day] > spanB[day] and baseline[day] < conversionline[day]
            print("sell trigger:", sell_trigger, self.coins)

            #buy_trigger
            if buy_trigger == True and self.cash > 0:
                available_funds = self.cash - (self.cash*fee_percentage)
                self.coins = float(self.coins + (available_funds/ ndo.iloc[day]))
                self.cash = 0 
            # sell trigger  
            elif sell_trigger == True and self.coins > 0:
                potential_funds= ndo.iloc[day] * self.coins
                funds_gained = potential_funds - (potential_funds*fee_percentage)
                self.cash = float(self.cash + funds_gained)
                self.coins = 0
            portfolio_value.append(self.cash)
            buy_signal.append(buy_trigger)
            sell_signal.append(sell_trigger)
                
        if self.coins > 0:
            self.cash = float(self.cash + (ndo.iloc[day] * self.coins))
            self.coins = 0   
            portfolio_value[len(portfolio_value)-1] = self.cash
        print(self.cash,self.coins)
        
        trade_signals.drop("spanA", axis =1, inplace =True)
        trade_signals.drop("spanB", axis =1, inplace =True)
        trade_signals.drop("Baseline", axis =1, inplace =True)
        trade_signals.drop("ConversionLine", axis =1, inplace =True)
        
        return pd.Series(ndo, name = "next_day_open"), pd.Series(buy_signal, name="buy_signal"), pd.Series(sell_signal, name="sell_signal")
    
        
main()