"""
CS7646 Summer 2021 Project 6 - Part1. Technical Indicators
Student Name: Renyu Zhang
GT User ID: rzhang605
GT ID: 903653510
"""

import datetime as dt
import pandas as pd
from util import get_data
import marketsimcode as mkt

# Georgia Tech user ID
def author():
    return "CarolineRYZ"

# calculate simple moving average (sma)
def my_sma(price_df, days):
    return (price_df.rolling(window=days, min_periods=days).mean())

# calculate price/SMA ratio
def price_sma(price_df, days):
    return (price_df / my_sma(price_df, days))

# calculate the standard deviation of daily returns
def volatility(price_df, days):
    return (price_df.rolling(window=days, min_periods=days).std())

# calculate Bollinger Bands (bb)
def top_bb(price_df, days):
    return (my_sma(price_df, days) + 2 * volatility(price_df, days))

def bottom_bb(price_df, days):
    return (my_sma(price_df, days) - 2 * volatility(price_df, days))

def bb_percent(price_df, days):
    return ((price_df - bottom_bb(price_df, days)) / (top_bb(price_df, days) - bottom_bb(price_df, days)))

# calculate momentum
def momentum(price_df, days):
    return (price_df / price_df.shift(days) - 1)

# calculate exponential moving average (EMA)
def ema(price_df, days=9):
    return price_df.ewm(span=days, adjust=False).mean()

def ema1(price_df, days=12):
    return price_df.ewm(span=days, adjust=False).mean()

def ema2(price_df, days=26):
    return price_df.ewm(span=days, adjust=False).mean()

def my_ema(price_df, days):
    return price_df.ewm(span=days, adjust=False).mean()

# calculate price/EMA ratio
def price_ema(price_df, days):
    return (price_df / my_ema(price_df, days))


# calculate moving average convergence/divergence using MACD(12,26,9)
def macd(price_df):
    macd_df = ema1(price_df, 12) - ema2(price_df, 26)
    return macd_df

def macd_signal(price_df):
    signal = ema(macd(price_df), 9)
    return signal

def test_indicators():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)

    # get stocks data and fill the missing data
    prices_all = get_data(symbols=["JPM"], dates=dates, addSPY=True, colname="Adj Close")  # automatically adds SPY

    prices_all.fillna(method="ffill")
    prices_all.fillna(method="bfill")
    prices_all /= prices_all.iloc[0]  # normalize stock prices
    prices_sym = prices_all["JPM"]  # only the symbol

    days = 15
    # indicator 1. Price/SMA
    sma = my_sma(prices_sym, days)
    priceSMA = price_sma(prices_sym, days)

    sma.plot()
    priceSMA.plot()
    prices_sym.plot()
    mkt.plotting("Time", "Price", "Indicator 1. Price/SMA Ratio", ["15-day SMA", "Price/SMA", "Normalized JPM"], "indicator1.png")

    # indicator 2. Bollinger Bands
    top_band = top_bb(prices_sym, days)
    bottom_band = bottom_bb(prices_sym, days)
    bbp = bb_percent(prices_sym, days)

    top_band.plot(color = "c")
    bottom_band.plot(color = "c")
    sma.plot(color = "r")
    prices_sym.plot(color = "y")
    mkt.plotting("Time", "Price", "Indicator 2.1 Bollinger Bands", ["Top Band", "Bottom Band", "SMA", "Normalized JPM"], "indicator2.1.png")

    bbp.plot()
    #top_band.plot()
    #bottom_band.plot()
    #sma.plot()
    prices_sym.plot()
    mkt.plotting("Time", "Price", "Indicator 2.2 Bollinger Bands %",
                 ["BB%", "Normalized JPM"], "indicator2.2.png")

    # indicator 3. Momentum
    mtm = momentum(prices_sym, days)

    mtm.plot()
    prices_sym.plot()
    mkt.plotting("Time", "Price", "Indicator 3. Momentum", ["Momentum", "Normalized JPM"], "indicator3.png")


    # indicator 4. Price/EMA
    ema12 = ema1(prices_sym, days=12)
    ema26 = ema2(prices_sym, days=26)
    emaRatio = price_ema(prices_sym, days)

    emaRatio.plot()
    ema12.plot()
    ema26.plot()
    prices_sym.plot(color = "m")
    mkt.plotting("Time", "Price", "Indicator 4. Price/EMA Ratio", ["Price/EMA", "12-day EMA", "26-day EMA", "Normalized JPM"], "indicator4.png")

    # indicator 5. MACD
    macdLine = macd(prices_sym) * 5
    signalLine = macd_signal(prices_sym) * 5

    macdLine.plot()
    signalLine.plot()
    prices_sym.plot()
    mkt.plotting("Time", "Price", "Indicator 5. MACD", ["MACD", "Signal", "Normalized JPM"], "indicator5.png")


if __name__ == "__main__":
    test_indicators()
