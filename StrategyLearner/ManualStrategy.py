"""
CS7646 Summer 2021 Project 8 - Implement Manual Rule-Based Trader
Student Name: Renyu Zhang
GT User ID: rzhang605
GT ID: 903653510
"""


import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import util as ut
import indicators as ind
import marketsimcode as mkt


def author():
    return "CarolineRYZ"


def testPolicy(symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    symbol = [symbol]
    dates = pd.date_range(sd, ed)

    # get stocks data and fill the missing data
    prices_all = ut.get_data(symbol, dates, addSPY=True, colname="Adj Close")  # automatically adds SPY

    prices_all.fillna(method="ffill")
    prices_all.fillna(method="bfill")
    prices_all /= prices_all.iloc[0]  # normalize stock prices
    prices_sym = prices_all[symbol]  # only the symbol

    days = 15

    smaRatio = ind.price_sma(prices_sym, days)
    bbPercent = ind.bb_percent(prices_sym, days)
    emaRatio = ind.price_ema(prices_sym, days)

    df_trades = prices_sym.copy()
    df_trades[symbol] = 0.0

    holdings = 0.0

    # long stock when price/sma < 0.95, bb% < 0, price/ema < 0.95, mtm < -0.25, ema < 0.95
    # short stock when price/sma > 1.05, bb% > 1, price/ema > 1.1, mtm > 0.25, ema > 1
    for i in range(len(prices_sym.index) - 1):
        sma = smaRatio.iloc[i, 0]
        bbp = bbPercent.iloc[i, 0]
        ema = emaRatio.iloc[i, 0]

        if(holdings == 0 and sma < 0.95 and bbp < 0 and ema < 0.95):
            df_trades.iloc[i, 0] = 1000
            holdings += 1000

        if (holdings == 0 and sma > 1.05 and bbp > 1 and ema > 1.1):
            df_trades.iloc[i, 0] = -1000
            holdings -= 1000

        if (holdings == 1000 and sma > 1.05 and bbp > 1 and ema > 1.1):
            df_trades.iloc[i, 0] = -2000
            holdings -= 2000

        if (holdings == -1000 and sma < 0.95 and bbp < 0 and ema < 0.95):
            df_trades.iloc[i, 0] = 2000
            holdings += 2000

    df_trades.dropna(inplace=True)

    return df_trades


def testManualStrategy(verbose=False):
    # in-sample period
    sym = "JPM"
    sdin = dt.datetime(2008, 1, 1)
    edin=dt.datetime(2009, 12, 31)
    sv=100000
    commission = 9.95
    impact = 0.005

    df_in = testPolicy(symbol=sym, sd=sdin, ed=edin, sv=sv)

    portvals_in = mkt.compute_portvals(df_in, start_val=sv)
    if isinstance(portvals_in, pd.DataFrame):
        portvals_in = portvals_in[portvals_in.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # get benchmark value
    benchmark_in = df_in.copy()
    benchmark_in[sym] = 0.0
    benchmark_in.iloc[0, 0] = 1000 # invest in 1000 shares of JPM and holding

    benchvals_in = mkt.compute_portvals(benchmark_in, start_val=sv, commission=commission, impact=impact)
    if isinstance(benchvals_in, pd.DataFrame):
        benchvals_in = benchvals_in[benchvals_in.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"


    # out-of-sample period
    sdout = dt.datetime(2010, 1, 1)
    edout=dt.datetime(2011, 12, 31)

    df_out = testPolicy(symbol=sym, sd=sdout, ed=edout, sv=sv)

    portvals_out = mkt.compute_portvals(df_out, start_val=sv)
    if isinstance(portvals_out, pd.DataFrame):
        portvals_out = portvals_out[portvals_out.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # get benchmark value
    benchmark_out = df_out.copy()
    benchmark_out[sym] = 0.0
    benchmark_out.iloc[0, 0] = 1000 # invest in 1000 shares of JPM and holding

    benchvals_out = mkt.compute_portvals(benchmark_out, start_val=sv, commission=commission, impact=impact)
    if isinstance(benchvals_out, pd.DataFrame):
        benchvals_out = benchvals_out[benchvals_out.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # get stats
    cum_ret_in, avg_daily_ret_in, std_daily_ret_in, sharpe_ratio_in = mkt.stats(portvals_in)
    cum_ret_bench_in, avg_daily_ret_bench_in, std_daily_ret_bench_in, sharpe_ratio_bench_in = mkt.stats(benchvals_in)
    cum_ret_out, avg_daily_ret_out, std_daily_ret_out, sharpe_ratio_out = mkt.stats(portvals_out)
    cum_ret_benchout, avg_daily_ret_benchout, std_daily_ret_benchout, sharpe_ratio_benchout = mkt.stats(benchvals_out)

    if verbose:
        # compare in-sample results
        print("Manual Strategy")
        print(f"In-sample period: {sdin} to {edin}")
        print()
        print(f"Sharpe Ratio of Manual Strategy: {sharpe_ratio_in:.4f}")
        print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_bench_in:.4f}")
        print()
        print(f"Cumulative Return of Manual Strategy: {cum_ret_in:.4f}")
        print(f"Cumulative Return of Benchmark : {cum_ret_bench_in:.4f}")
        print()
        print(f"Standard Deviation of Manual Strategy: {std_daily_ret_in:.4f}")
        print(f"Standard Deviation of Benchmark : {std_daily_ret_bench_in:.4f}")
        print()
        print(f"Average Daily Return of Manual Strategy: {avg_daily_ret_in:.4f}")
        print(f"Average Daily Return of Benchmark : {avg_daily_ret_bench_in:.4f}")
        print()
        print(f"Final Manual Strategy Value: ${portvals_in[-1]:,.2f}")
        print(f"Final Benchmark Value: ${benchvals_in[-1]:,.2f}")
        print()

        # compare out-of-sample results
        print("Manual Strategy")
        print(f"0ut-of-sample period: {sdout} to {edout}")
        print()
        print(f"Sharpe Ratio of Manual Strategy: {sharpe_ratio_out:.4f}")
        print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_benchout:.4f}")
        print()
        print(f"Cumulative Return of Manual Strategy: {cum_ret_out:.4f}")
        print(f"Cumulative Return of Benchmark : {cum_ret_benchout:.4f}")
        print()
        print(f"Standard Deviation of Manual Strategy: {std_daily_ret_out:.4f}")
        print(f"Standard Deviation of Benchmark : {std_daily_ret_benchout:.4f}")
        print()
        print(f"Average Daily Return of Manual Strategy: {avg_daily_ret_out:.4f}")
        print(f"Average Daily Return of Benchmark : {avg_daily_ret_benchout:.4f}")
        print()
        print(f"Final Manual Strategy Value: ${portvals_out[-1]:,.2f}")
        print(f"Final Benchmark Value: ${benchvals_out[-1]:,.2f}")
        print("\n")

    # create charts
    normalized_port_in = portvals_in / portvals_in.iloc[0]
    normalized_bench_in = benchvals_in / benchvals_in.iloc[0]
    normalized_port_out = portvals_out / portvals_out.iloc[0]
    normalized_bench_out = benchvals_out / benchvals_out.iloc[0]

    # get buy and sell tradings
    df_in_buy = df_in[(df_in.T > 0).any()]
    df_in_sell = df_in[(df_in.T < 0).any()]
    df_out_buy = df_out[(df_out.T > 0).any()]
    df_out_sell = df_out[(df_out.T < 0).any()]
    #print("\n\n", df_in_buy, "\n\n", df_in_sell, "\n\n", df_out_buy, "\n\n", df_out_sell)

    # In-Sample Manual Strategy v.s. Benchmark
    normalized_bench_in.plot(color = "g")
    normalized_port_in.plot(color = "r")
    xmin_in, xmax_in, ymin_in, ymax_in = plt.axis()
    plt.vlines(df_in_sell.index.tolist(), ymin_in, ymax_in, color = "k")
    plt.vlines(df_in_buy.index.tolist(), ymin_in, ymax_in, color="b")
    mkt.plotting("Time", "Normalized Value", "In-Sample Manual Strategy v.s. Benchmark",
                 ["Benchmark", "Manual Strategy"], "in_sample.png")

    # Out-of-Sample Manual Strategy v.s. Benchmark
    normalized_bench_out.plot(color = "g")
    normalized_port_out.plot(color = "r")
    xmin_out, xmax_out, ymin_out, ymax_out = plt.axis()
    plt.vlines(df_out_sell.index.tolist(), ymin_out, ymax_out, color="k")
    plt.vlines(df_out_buy.index.tolist(), ymin_out, ymax_out, color="b")
    mkt.plotting("Time", "Normalized Value", "Out-of-Sample Manual Strategy v.s. Benchmark",
                 ["Benchmark", "Manual Strategy"], "out_of_sample.png")


if __name__ == "__main__":
    testManualStrategy()

