import ta
import ccxt
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volume import ChaikinMoneyFlowIndicator
import pandas as pd
import numpy as np


'''Control the whole process of generting parameters and running the bot, storing and evaluating the output'''    
def main():
    INIT_MONEY = 100.0
    SRSI_BUY = 0.2
    SRSI_SELL= 0.8
    
    
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

    bot = TradeBot(SRSI_BUY, SRSI_SELL, INIT_MONEY,14, 50, 50,bitcoin_indicators, srsi_data )
    bot.execute_period()
    print("final dump", bot.cash)
    #nsga3 = NSGAIII()
    

class TradeBot:
    def __init__ (self, SRSI_buy, SRSI_sell, Money, SRSI_window, SRSI_smooth1, SRSI_smooth2, TOHCLV_data, srsi_data):
        print(type(Money))
        self.buy_trigger = SRSI_buy
        self.sell_trigger =SRSI_sell
        self.cash = Money
        self.period = SRSI_window
        self.smooth1 = SRSI_smooth1
        self.smooth2 = SRSI_smooth2
        self.tohclv = TOHCLV_data
        self.srsidata = srsi_data
        # set initial bprint(type(Money))itcoins to zero
        self.coins = 0
        print("buy_trig: ", self.buy_trigger)
        print("sell_trig: ", self.sell_trigger)
        print(type(Money))
    #def __str__(self):
    #   rep ="Funds at hand: {0}".format(self.cash)
    #    return rep
        
        
    def execute_period(self):
        buy_trigger_status = False
        sell_trigger_status = False
        
        for day in range(len(self.tohclv)):
            srsi_reading=self.srsidata[day]
            
            
            #price_array = self.TOHCLV.to_array()
            current_price = (self.tohclv['open'] + self.tohclv['close'] + self.tohclv['high'] + self.tohclv['low'])/4
            #print("current price", current_price[day])
            print(day, " srsi", srsi_reading, "bot: ",self.cash, " bitcoins: ", self.coins,"current_price", current_price[day])
            # sell trigger
            if srsi_reading >= self.sell_trigger:
                sell_trigger_status = True
                buy_trigger_status = False
            
            #buy trigger
            elif srsi_reading <= self.buy_trigger:
                buy_trigger_status = True
                sell_trigger_status = False
            
            if sell_trigger_status == True and self.coins > 0:
                potential_funds= current_price[day] * self.coins
                funds_gained = potential_funds - (potential_funds/50)
                self.cash = float(self.cash + funds_gained)
                self.coins = 0
            elif buy_trigger_status == True:
                available_funds = self.cash - (self.cash/50)
                self.coins = float(self.coins + (available_funds/ current_price[day]))
                self.cash = 0 #float((self.cash - (self.cash//current_price[day]) * current_price[day]))
        
        if self.coins > 0:
            self.cash = float(self.cash + (current_price[day] * self.coins))
            self.coins = 0
                
            
class NSGAIII:
    def __init__(self, Inpop, generations, SRSI_buy, SRSI_sell,TOHCLV, srsi_data, srefpoints=None):
        self.newpop_s= np.zeros([len(Inpop),len(Inpop[:,0])])
        self.pop=Inpop
        self.generations= generations
        self.srsi_buy = SRSI_buy
        self.srsi_sell = SRSI_sell
        self.refpoints= srefpoints
        self.tohclv = TOHCLV
        self.srsi_data=srsi_data
        
            
    def dominates (self, x1, x2):
        num_data_points = 100
        rnd=np.random.default_rng()
        start=rnd.integers(low=0, high = 720 - num_data_points)
        stop = start + num_data_points
    
        botx1 = TradeBot(self.srsi_buy, self.srsi_sell, x1[0], x1[1], x1[2], self.tohclv[start:stop], self.srsi_data[start:stop])
        botx2 = TradeBot(self.srsi_buy, self.srsi_sell, x2[0], x2[1], x2[2], self.tohclv[start:stop], self.srsi_data[start:stop])
        
        return botx1 >= botx2
    
    def nondominated_sort(self, pop):
        for x_index in range(pop.to_array().shape[0]-1):
            x1 = pop[x_index]
            x2 = pop[x_index+1]
            result=NSGAIII.dominates(x1, x2)
            
            if result == True:
                continue
            else:
                pop[x_index], pop[x_index + 1] = pop[x_index + 1], pop[x_index]
        
    def mutate(self, pop):
        rnd = np.random.default_rng()
        window = 50
        smooth1 = 20
        smooth2 = 20
        
        rndparameterlist = [window,smooth1, smooth2]
        rndparameterchoice = rnd.integers(low=0, high = 2)
        newparametervalue = rnd.integers(low=0, high = rndparameterlist[rndparameterchoice])
        
        # choose random individual to mutate (not the best)

        indtomut = rnd.intergers(low=1, high = len(pop))
        
        pop.iloc[indtomut, rndparameterchoice] = newparametervalue
        return pop
        
        
        
        
    
    def crossover(self):
        pass
    
    def recombinant(self):
        self
        
    
main()          

#print(EA.dominates([20,2,2], [14,2,2]))
