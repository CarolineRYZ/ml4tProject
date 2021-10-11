"""
CS7646 Summer 2021 Project 8 - Experiment 1
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


def testStrategy(verbose=False):
    np.random.seed(903653510)  # use my GT ID as the numeric value for the random seed
    sym = "JPM"
    sv = 100000
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    # 1. Strategy Learner: Random Forest
    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95) # instantiate the learner object
    learner.add_evidence(symbol=sym, sd=start_date, ed=end_date, sv=sv) # training

    trades1 = learner.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=sv) # in-sample test

    portvals1 = mkt.compute_portvals(trades1, start_val=sv)
    if isinstance(portvals1, pd.DataFrame):
        portvals1 = portvals1[portvals1.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # 2. Manual Strategy
    trades2 = ms.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=sv)

    portvals2 = mkt.compute_portvals(trades2, start_val=sv)
    if isinstance(portvals2, pd.DataFrame):
        portvals2 = portvals2[portvals2.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # 3. Benchmark
    benchmark = trades2.copy()
    benchmark[[sym]] = 0.0
    benchmark.iloc[0, 0] = 1000  # invest in 1000 shares of JPM and holding

    benchvals = mkt.compute_portvals(benchmark, start_val=sv)
    if isinstance(benchvals, pd.DataFrame):
        benchvals = benchvals[benchvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Create chart
    normalized_port1 = portvals1 / portvals1.iloc[0]
    normalized_port2 = portvals2 / portvals1.iloc[0]
    normalized_bench = benchvals / benchvals.iloc[0]

    normalized_bench.plot(color="g")
    normalized_port1.plot(color="y")
    normalized_port2.plot(color="r")
    mkt.plotting("Time", "Normalized Value", "Experiment 1", ["Benchmark", "Strategy Learner", "Manual Strategy"],"experiment1.png")

    # calculate stats
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = mkt.stats(benchvals)
    cum_ret1, avg_daily_ret1, std_daily_ret1, sharpe_ratio1 = mkt.stats(portvals1)
    cum_ret2, avg_daily_ret2, std_daily_ret2, sharpe_ratio2 = mkt.stats(portvals2)

    if verbose:
        print("Experiment 1.")
        print(f"Date Range: {start_date} to {end_date}")
        print()
        print(f"Sharpe Ratio of Strategy Learner: {sharpe_ratio1:.4f}")
        print(f"Sharpe Ratio of Manual Strategy: {sharpe_ratio2:.4f}")
        print(f"Sharpe Ratio of Benchmark: {sharpe_ratio_benchmark:.4f}")
        print()
        print(f"Cumulative Return of Strategy Learner: {cum_ret1:.4f}")
        print(f"Cumulative Return of Manual Strategy: {cum_ret2:.4f}")
        print(f"Cumulative Return of Benchmark: {cum_ret_benchmark:.4f}")
        print()
        print(f"Standard Deviation of Strategy Learner: {std_daily_ret1:.4f}")
        print(f"Standard Deviation of Manual Strategy: {std_daily_ret2:.4f}")
        print(f"Standard Deviation of Benchmark: {std_daily_ret_benchmark:.4f}")
        print()
        print(f"Average Daily Return of Strategy Learner: {avg_daily_ret1:.4f}")
        print(f"Average Daily Return of Manual Strategy: {avg_daily_ret2:.4f}")
        print(f"Average Daily Return of Benchmark: {avg_daily_ret_benchmark:.4f}")
        print()
        print(f"Final Strategy Learner's Portfolio Value: ${portvals1[-1]:,.2f}")
        print(f"Final Manual Strategy's Portfolio Value: ${portvals2[-1]:,.2f}")
        print(f"Final Benchmark Value: ${benchvals[-1]:,.2f}")
        print("\n")


if __name__ == "__main__":
    testStrategy()
