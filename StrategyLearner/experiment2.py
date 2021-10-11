"""
CS7646 Summer 2021 Project 8 - Experiment 2
Student Name: Renyu Zhang
GT User ID: rzhang605
GT ID: 903653510
"""


import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import marketsimcode as mkt
import indicators as ind
import ManualStrategy as ms
import StrategyLearner as sl


def author():
    return "CarolineRYZ"


def testImpact(verbose=False):

    # use my GT ID as the numeric value for the random seed
    np.random.seed(903653510)

    # in-sample period
    sym = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0.0
    impact1 = 0.0
    impact2 = 0.005
    impact3 = 0.05

    # 1. Strategy Learner: impact1 = 0.0
    learner1 = sl.StrategyLearner(verbose=False, impact=impact1, commission=commission)  # instantiate the learner object
    learner1.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=sv)  # training

    trades1 = learner1.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=sv)  # in-sample test
    portvals1 = mkt.compute_portvals(trades1, start_val=sv)
    if isinstance(portvals1, pd.DataFrame):
        portvals1 = portvals1[portvals1.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # 2. Strategy Learner: impact2 = 0.005
    learner2 = sl.StrategyLearner(verbose=False, impact=impact2, commission=commission)  # instantiate the learner object
    learner2.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=sv)  # training

    trades2 = learner2.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=sv)  # in-sample test
    portvals2 = mkt.compute_portvals(trades2, start_val=sv)
    if isinstance(portvals2, pd.DataFrame):
        portvals2 = portvals2[portvals2.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # 3. Strategy Learner: impact3 = 0.05
    learner3 = sl.StrategyLearner(verbose=False, impact=impact3, commission=commission)  # instantiate the learner object
    learner3.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=sv)  # training

    trades3 = learner3.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=sv)  # in-sample test
    portvals3 = mkt.compute_portvals(trades3, start_val=sv)
    if isinstance(portvals3, pd.DataFrame):
        portvals3 = portvals3[portvals3.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Create chart
    normalized_port1 = portvals1 / portvals1.iloc[0]
    normalized_port2 = portvals2 / portvals1.iloc[0]
    normalized_port3 = portvals3 / portvals1.iloc[0]

    normalized_port1.plot(color="y")
    normalized_port2.plot(color="r")
    normalized_port3.plot(color="c")
    mkt.plotting("Time", "Normalized Value", "Experiment 2", ["impact = 0.0", "impact = 0.005", "impact = 0.05"],"experiment2.png")

    # calculate stats
    cum_ret1, avg_daily_ret1, std_daily_ret1, sharpe_ratio1 = mkt.stats(portvals1)
    cum_ret2, avg_daily_ret2, std_daily_ret2, sharpe_ratio2 = mkt.stats(portvals2)
    cum_ret3, avg_daily_ret3, std_daily_ret3, sharpe_ratio3 = mkt.stats(portvals3)

    if verbose:
        print("Experiment 2.")
        print(f"Date Range: {start_date} to {end_date}")
        print()
        print(f"Sharpe Ratio of Strategy Learner (impact = 0.0): {sharpe_ratio1:.4f}")
        print(f"Sharpe Ratio of Strategy Learner (impact = 0.005): {sharpe_ratio2:.4f}")
        print(f"Sharpe Ratio of Strategy Learner (impact = 0.05): {sharpe_ratio3:.4f}")
        print()
        print(f"Cumulative Return of Strategy Learner (impact = 0.0): {cum_ret1:.4f}")
        print(f"Cumulative Return of Strategy Learner (impact = 0.005): {cum_ret2:.4f}")
        print(f"Cumulative Return of Strategy Learner (impact = 0.05): {cum_ret3:.4f}")
        print()
        print(f"Standard Deviation of Strategy Learner (impact = 0.0): {std_daily_ret1:.4f}")
        print(f"Standard Deviation of Strategy Learner (impact = 0.005): {std_daily_ret2:.4f}")
        print(f"Standard Deviation of Strategy Learner (impact = 0.05): {std_daily_ret3:.4f}")
        print()
        print(f"Average Daily Return of Strategy Learner (impact = 0.0): {avg_daily_ret1:.4f}")
        print(f"Average Daily Return of Strategy Learner (impact = 0.005): {avg_daily_ret2:.4f}")
        print(f"Average Daily Return of Strategy Learner (impact = 0.05): {avg_daily_ret3:.4f}")
        print()
        print(f"Final Portfolio Value of Strategy Learner (impact = 0.0): ${portvals1[-1]:,.2f}")
        print(f"Final Portfolio Value of Strategy Learner (impact = 0.005): ${portvals2[-1]:,.2f}")
        print(f"Final Portfolio Value of Strategy Learner (impact = 0.05): ${portvals3[-1]:,.2f}")
        print()


if __name__ == "__main__":
    testImpact()
