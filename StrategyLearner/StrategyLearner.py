""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Renyu Zhang (replace with your name)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: rzhang605 (replace with your User ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 903653510 (replace with your GT ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt
import indicators as ind
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import util as ut
import BagLearner as bl
import RTLearner as rl
import numpy as np

class StrategyLearner(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type impact: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type commission: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # constructor  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.verbose = verbose  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.impact = impact  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self.commission = commission
        self.days = 10
        # instantiate a Baglearner with random trees (i.e. random forest learner)
        self.learner = bl.BagLearner(learner = rl.RTLearner, kwargs = {"leaf_size": 5}, bags = 10, boost = False, verbose = False)

    # my GT user ID
    def author(self):
        return "CarolineRYZ"
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this method should create a RTLearner, and train it for trading
    def add_evidence(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=100000
    ):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type symbol: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sv: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        # add your code to do learning here  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        # example usage of the old backward compatible util function  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        syms = [symbol]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        dates = pd.date_range(sd, ed)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        prices_sym = prices_all[syms]  # only portfolio symbols
        if self.verbose:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            print(prices_sym)

        prices_sym /= prices_sym.iloc[0]  # normalize portfolio prices

        # get indicators as training features
        xtrain = self.indicators(prices_sym)

        # create ytrain for indicators (xtrain) using rolling forward cross validation
        # set market variance to 0.01
        Ndays = 10
        daily_ret = prices_sym / prices_sym.shift(Ndays) - 1
        daily_ret = daily_ret[Ndays:]

        ytrain = daily_ret.copy()

        for i in range(ytrain.shape[0]):

            if daily_ret.iloc[i,0] > (0.01 + self.impact):
                ytrain.iloc[i,0] = 1 # long

            elif daily_ret.iloc[i,0] < (-0.01 - self.impact):
                ytrain.iloc[i,0] = -1 # short

            else:
                ytrain.iloc[i, 0] = 0  # cash

        ytrain = np.array(ytrain)

        self.learner.add_evidence(xtrain, ytrain)

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def testPolicy(  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        self,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol="IBM",  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sv=100000
    ):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type symbol: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sv: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        # here we build a fake set of trades  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        # your code should return the same sort of data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        dates = pd.date_range(sd, ed)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        trades = prices_all[[symbol]]  # only portfolio symbols

        trades /= trades.iloc[0]  # normalize portfolio prices

        # get indicators as training features
        xtest = self.indicators(trades)

        # predict y
        ytest = self.learner.query(xtest)

        df_trades = trades.copy()
        df_trades[symbol] = 0.0

        holdings = 0.0

        for i in range(ytest.shape[0]):

            if (holdings == 0 and ytest[i] == 1):
                df_trades.iloc[i, 0] = 1000
                holdings += 1000

            if (holdings == 0 and ytest[i] == -1):
                df_trades.iloc[i, 0] = -1000
                holdings -= 1000

            if (holdings == 1000 and ytest[i] == -1):
                df_trades.iloc[i, 0] = -2000
                holdings -= 2000

            if (holdings == -1000 and ytest[i] == 1):
                df_trades.iloc[i, 0] = 2000
                holdings += 2000

        if self.verbose:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            print(type(trades))  # it better be a DataFrame!  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        if self.verbose:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            print(trades)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        if self.verbose:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            print(prices_all)

        return df_trades

    def indicators(self, prices):
        # get indicators as training features
        smaRatio = ind.price_sma(prices, 5)
        bbPercent = ind.bb_percent(prices, 20)
        emaRatio = ind.price_ema(prices, 5)

        xdata = pd.concat((smaRatio, bbPercent, emaRatio), axis=1)
        xdata.fillna(0, inplace=True)
        xdata = xdata[self.days :]

        return xdata.values

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print("One does not simply think up a strategy")
