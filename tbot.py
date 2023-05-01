#import ta
import ccxt
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD
import pandas as pd
import numpy as np
import copy


'''Control the whole process of generting parameters and running the bot, storing and evaluating the output'''    
def main():
    rnd = np.random.default_rng()
    
    INIT_MONEY = 100.0
    RSI_BUY = 30
    RSI_SELL = 70
    SRSI_BUY = 0.2
    SRSI_SELL= 0.8
    NUM_INIT_VECTORS = 6
    NUM_GENERATIONS = 10
    
    #initialise data
    kraken_exchange = ccxt.kraken()
    kraken_market = kraken_exchange.load_markets()
    #for mkt in kraken_market:
        #print(mkt)
        
    bitcoin_data = kraken_exchange.fetch_ohlcv("BTC/AUD", timeframe="1d", limit = 720)
    bitcoin_indicators = pd.DataFrame(bitcoin_data, columns = ["timestamp","open", "high", "low", "close", "volume"])
    
    srsi_ind = StochRSIIndicator(bitcoin_indicators['close'])
    srsi_data= srsi_ind.stochrsi()
    rsi_ind = RSIIndicator(bitcoin_indicators['close'])
    rsi_data =rsi_ind.rsi()
    macd_ind = MACD(bitcoin_indicators['close'])
    macd_line = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()
    
    print("macdl",macd_line)
    print("MSignal",macd_sig)
    
    
    # generate initial population vectors
    vectors=np.array([])
    for vect in range(NUM_INIT_VECTORS):
        
        print("vs",vectors.size)
        srsi_win = rnd.integers(low=1, high=50)
        srsi_smooth1 = rnd.integers(low=1, high=20)
        srsi_smooth2  = rnd.integers(low=1, high=20) 
        
        rsi_window  = rnd.integers(low=1, high=30)   
        
        macd_window_signal  = rnd.integers(low=1, high=10)
        macd_window_slow  = rnd.integers(low=macd_window_signal , high=20)
        macd_window_fast  = rnd.integers(low=macd_window_signal, high=50) 
        
        
        
        vect = np.array([srsi_win,srsi_smooth1,srsi_smooth2,rsi_window ,macd_window_slow,macd_window_fast, macd_window_signal])
        print(vect,vectors)
        if vectors.size == 0:
            print("here")
            vectors=np.append(vectors, vect, axis=0)
        else:
            print("made it here")
            vectors=np.concatenate([vectors, vect], axis=0)
    vectors=vectors.reshape(NUM_INIT_VECTORS,vect.size).astype(int)
    print("vectors", vectors)                 
    
    evopop = EA(vectors, NUM_GENERATIONS , RSI_BUY, RSI_SELL, SRSI_BUY, SRSI_SELL, INIT_MONEY, bitcoin_indicators, bitcoin_indicators['close'], hprefpoints=None) # macd_line, macd_sig ,rsi_data, srsi_data,
    print(evopop.pop)
    epop =evopop.pop.copy()
    evopop.nondominated_sort()
    print("original pop: ", epop)
    print("final pop: ", evopop.pop)
    eopo1 =evopop.pop.copy()
    evopop.mutate()
    print("mutated", evopop.pop)
    
    return
    bot = TradeBot(RSI_BUY, RSI_SELL, SRSI_BUY, SRSI_SELL, INIT_MONEY,14, 50, 50,bitcoin_indicators, macd_line, macd_sig, rsi_data, srsi_data )
    bot.execute_period()
    print("final dump", bot.cash)
    #nsga3 = NSGAIII()
    
def reset_indicators(bitcoin_indicator_close, SRSI_window, SRSI_smooth1, SRSI_smooth2,rsi_window, macd_windowfast, macd_windowslow, macdsignal, start, stop):
    srsi_ind = StochRSIIndicator(bitcoin_indicator_close, window=SRSI_window, smooth1=SRSI_smooth1, smooth2=SRSI_smooth2)
    srsi_data= srsi_ind.stochrsi()
    rsi_ind = RSIIndicator(bitcoin_indicator_close,window=rsi_window)
    rsi_data =rsi_ind.rsi()
    macd_ind = MACD(bitcoin_indicator_close,window_fast=macd_windowfast, window_slow=macd_windowslow, window_sign=macdsignal)
    macd_line = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()
    
    return srsi_data[start:stop], rsi_data[start:stop], macd_line[start:stop], macd_sig[start:stop]

class TradeBot:
    def __init__ (self, RSI_buy, RSI_sell, SRSI_buy, SRSI_sell, Money, SRSI_window, SRSI_smooth1, SRSI_smooth2,RSI_window, MACD_windowfast, MACD_windowslow, MACD_signal, TOHCLV_data, start, stop, bitcoinclose): #macdl_data, macds_data,rsi_data, srsi_data,
        self.rsibuy_trigger=RSI_buy
        self.rsisell_trigger=RSI_sell
        self.srsibuy_trigger = SRSI_buy
        self.srsisell_trigger =SRSI_sell
        self.cash = Money
        self.tohclv = TOHCLV_data
        self.coins = 0
        self.srsi_window = SRSI_window
        self.srsi_smooth1 = SRSI_smooth1
        self.srsi_smooth2 = SRSI_smooth2
        self.rsi_window = RSI_window
        self.macd_windowfast = MACD_windowfast
        self.macd_windowslow = MACD_windowslow
        self.signal = MACD_signal
        self.btclose = bitcoinclose
        
        
        srsi_data, rsi_data, macd_line, macd_sig= reset_indicators(bitcoinclose, SRSI_window, SRSI_smooth1, SRSI_smooth2, RSI_window, MACD_windowfast, MACD_windowslow, MACD_signal,start,stop )
        
        self.rsidata = rsi_data
        self.srsidata = srsi_data
        self.macd_line_data = macd_line
        self.macd_signal_data = macd_sig
        
        
    def __str__(self):
        rep ="Funds at hand: {0}".format(self.cash)
        return rep
        
        
    def execute_period(self):
        buy_trigger_status = False
        sell_trigger_status = False
        previous_transaction_day = -100
        
        
        
        for day in range(len(self.tohclv)):
            print("day",day)
            macd_line_reading=self.macd_line_data.iloc[day]
            macd_signal_reading = self.macd_signal_data.iloc[day]
            rsi_reading=self.rsidata.iloc[day]
            srsi_reading=self.srsidata.iloc[day]
            
            
            #price_array = self.TOHCLV.to_array()
            current_price = (self.tohclv['open'] + self.tohclv['close'] + self.tohclv['high'] + self.tohclv['low'])/4
            #print("current price", current_price[day])
            print(day, " srsi", srsi_reading, "bot: ",self.cash, " bitcoins: ", self.coins,"current_price", current_price.iloc[day])
            # sell trigger
            if macd_line_reading > 0 and macd_line_reading > macd_signal_reading and srsi_reading >= self.srsisell_trigger and rsi_reading > self.rsisell_trigger and ((previous_transaction_day + 1) != day): 
                sell_trigger_status = True
                buy_trigger_status = False
                previous_transaction_day = day
                
            #buy trigger
            elif macd_line_reading < 0 and macd_line_reading < macd_signal_reading and srsi_reading <= self.srsibuy_trigger and rsi_reading < self.rsibuy_trigger and ((previous_transaction_day + 1) != day):
                buy_trigger_status = True
                sell_trigger_status = False
                previous_transaction_day = day
            
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
        
                
            
class EA:
    def __init__(self, Inpop, generations, RSI_buy, RSI_sell, SRSI_buy, SRSI_sell, Money, TOHCLV_data,  bitcoin_indicator_close, hprefpoints=None):#macdl_data, macds_data,rsi_data, srsi_data,
        self.newpop_s= np.zeros([len(Inpop),len(Inpop[:,0])])
        self.pop = Inpop
        self.generations= generations
        self.rsi_buy = RSI_buy
        self.rsi_sell = RSI_sell
        self.srsi_buy = SRSI_buy
        self.srsi_sell = SRSI_sell
        self.money = Money
        self.refpoints = hprefpoints
        self.tohclv = TOHCLV_data
        self.refpoints = hprefpoints
        self.btcindclose = bitcoin_indicator_close
        
        '''
        self.macdldata = macdl_data
        self.macdsdata=macds_data
        self.rsidata = rsi_data
        self.srsidata = srsi_data
        '''
        
    def rounds(self, rounds):
        
        for round in range(rounds):
            self.nondominated_sort()
            self.mutate()
            self.crossover()            
            
    def dominates (self, x1, x2):
        num_data_points = 200
        rnd=np.random.default_rng()
        start=rnd.integers(low=0, high = 720 - num_data_points)
        stop = start + num_data_points
    
        #def __init__ (self, RSI_buy, RSI_sell, SRSI_buy, SRSI_sell, Money, SRSI_window, SRSI_smooth1, SRSI_smooth2, TOHCLV_data, macdl_data, macds_data,rsi_data, srsi_data):
        #  def __init__ (self, RSI_buy, RSI_sell, SRSI_buy, SRSI_sell, Money, SRSI_window, SRSI_smooth1, SRSI_smooth2,rsi_window, macd_windowfast, macd_windowslow, macdsignal, TOHCLV_data, macdl_data, macds_data,rsi_data, srsi_data):
        #
        botx1 = TradeBot(self.rsi_buy, self.rsi_sell, self.srsi_buy, self.srsi_sell, self.money,x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], self.tohclv[start:stop],start, stop,self.btcindclose) #self.macdldata[start:stop], self.macdsdata[start:stop],self.rsidata[start:stop], self.srsidata[start:stop]
        botx2 = TradeBot(self.rsi_buy, self.rsi_sell, self.srsi_buy, self.srsi_sell, self.money,x2[0], x2[1], x2[2], x2[3], x2[4], x2[5], x2[6], self.tohclv[start:stop], start, stop, self.btcindclose) # self.macdldata[start:stop], self.macdsdata[start:stop],self.rsidata[start:stop], self.srsidata[start:stop],
        
        botx1.execute_period()
        botx2.execute_period()
        
        print("bot cash", botx1.cash , botx2.cash)
        return botx1.cash < botx2.cash
    
    def nondominated_sort(self):
        for x_index in range(self.pop.shape[0]-1):
            x1 = self.pop[x_index]
            x2 = self.pop[x_index+1]
            result=self.dominates(x1, x2)
            
            if result == True:
                print("making it to here")
                self.pop[[x_index, x_index+1]] = self.pop[[x_index + 1, x_index]]
            else:
                continue
                
    def mutate(self):
        rnd = np.random.default_rng()
        srsi_window = 50
        srsi_smooth1 = 20
        srsi_smooth2 = 20
        rsi_window = 30
        macd_windowfast = 50
        macd_windowslow = 20
        macd_windowsignal = 10
        
        rndparameterlist = [srsi_window, srsi_smooth1, srsi_smooth2, rsi_window, macd_windowfast, macd_windowslow, macd_windowsignal]
        rndparameterchoice = rnd.integers(low=0, high = 6)
        newparametervalue = rnd.integers(low=1, high = rndparameterlist[rndparameterchoice])
        
        # choose random individual to mutate (not the best)

        indtomut = rnd.integers(low=1, high = len(self.pop))
        
        self.pop[indtomut, rndparameterchoice] = newparametervalue
        
        
        
    def crossover(self):
        rnd = np.random.default_rng()
        
        for index in range(0, len(self.pop) -1, 2):
            crossoverpnt = rnd.integers(low=1, high = len(self.pop.iloc[0]))
            self.pop.iloc[index], self.pop.iloc[index + 1]= np.concatenate([self.pop.iloc[index, 0:crossoverpnt],self.pop.iloc[index+1, crossoverpnt:len(self.pop.iloc[index + 1]) +1]], axis=0),np
                                                                            
        np.concatenate([self.pop.iloc[index, 0:crossoverpnt],self.pop.iloc[index+1, crossoverpnt:len(self.pop.iloc[index + 1]) +1]], axis=0), np.concatenate([self.pop.iloc[index+1, 0:crossoverpnt],self.pop.iloc[index, crossoverpnt:len(self.pop.iloc[index + 1]) +1]], axis=0)
    
    def recombinant(self):
        self
        
    
main()          

#print(EA.dominates([20,2,2], [14,2,2]))
