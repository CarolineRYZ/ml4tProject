""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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

import pandas as pd
import numpy as np
import util as ut
import matplotlib.pyplot as plt

def author():
    return "CarolineRYZ"


def compute_portvals(  		  	   		   	 			  		 			 	 	 		 		 	
    orders_df,
    start_val=1000000,  		  	   		   	 			  		 			 	 	 		 		 	
    commission=9.95,
    impact=0.005,
):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param orders_file: Path of the order file or the file object  		  	   		   	 			  		 			 	 	 		 		 	
    :type orders_file: str or file object  		  	   		   	 			  		 			 	 	 		 		 	
    :param start_val: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is the function the autograder will call to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 			  		 			 	 	 		 		 	
    # code should work correctly with either input  		  	   		   	 			  		 			 	 	 		 		 	
    # TODO: Your code here

    # get dates
    orders_df1 = orders_df.sort_index()
    sd = orders_df1.index[0]
    ed = orders_df1.index[-1]
    dates = pd.date_range(sd, ed)

    # get symbols
    symbols = orders_df1.columns.unique().tolist()

    # get stocks data and fill the missing data
    prices_all = ut.get_data(symbols, dates, addSPY=True, colname="Adj Close")  # automatically adds SPY

    prices_all.fillna(method="ffill")
    prices_all.fillna(method="bfill")

    prices_df2 = prices_all[symbols]  # only portfolio symbols

    prices_df2["cash_balance"] = 1.0 # add the column for later df multiplication

    # df to keep track of stock shares in the portfolio and daily cash balance
    changes_df3 = orders_df1
    changes_df3["cash_balance"] = 0.0

    for index, row in orders_df1.iterrows():
        shares = row[symbols[0]]
        price = prices_df2.loc[index, symbols[0]]

        # commission and impact for a trade
        changes_df3.loc[index, "cash_balance"] -= commission

        changes_df3.loc[index, "cash_balance"] -= shares * price * (1 + impact)

    positions_df4 = pd.DataFrame(index=prices_df2.index, columns=symbols)
    positions_df4.loc[:, :] = 0.0  # df initialization
    positions_df4["cash_balance"] = 0.0

    positions_df4.iloc[0, :] = changes_df3.iloc[0, :]  # initialize the first row of df4
    positions_df4.iloc[0, -1] += start_val  # add the start value of the portfolio to the df

    # update the daily positions of the portfolio with the changes
    for index in range(len(positions_df4.index) - 1):
        positions_df4.iloc[index + 1, :] = positions_df4.iloc[index, :] + changes_df3.iloc[index + 1, :]

    portvals = positions_df4.mul(prices_df2)

    portvals = portvals.sum(axis=1)

    return portvals

def stats(portvals):
    cum_ret = (portvals[-1] / portvals[0]) - 1
    daily_ret = (portvals / portvals.shift(1)) - 1  # compute daily returns for row 1 onwards
    daily_ret = daily_ret[1:]
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = np.std(daily_ret)
    sharpe_ratio = np.sqrt(252) * (avg_daily_ret / std_daily_ret)

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def plotting(x, y, title, legend, save):
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(save)
    plt.clf()
