import pandas as pd
import logging

def rolling_corr(df: pd.DataFrame, winsz: int = 505, minp: int = 375) -> pd.DataFrame:
    """
    Rolling correlation of each column of df against the last column of df
    :param df: pandas DataFrame with columns of numeric type, e.g. float to allow NaNs
    :param winsz: rolling window size
    :param minp: minimum number of observations in the window (overlapping in time) for pairwise correlation;
        in other words, complete pairs (two observations from two time series at a given time)
        if less than minp, NaNs are returned
    :return: pandas DataFrame with pairwise rolling correlation of each column of df
        against the last column of df
    """
    return (
        df.
        rolling(window=winsz, min_periods=minp).
        corr(df.iloc[:, -1])  # pairwise rolling correlation of stock vs market returns
    )  # in prod code, it's best to rely on names (unless for performance reasons) to reduce risk of bugs


def fill_firstNaN_ingaps(df: pd.DataFrame, val: float = 0.0) -> pd.DataFrame:
    """
    Fill NaNs in each column of df with val but only if there are no NaNs before filling

    :param df: pandas DataFrame with columns of numeric type, e.g. float to allow NaNs
    :param val: value to fill NaNs with (defaults 0.0, which is less interfering for stock returns than
        previous observation, which is more appropriate for price level, or mean or median, which add
        look-ahead bias). can also be e.g. risk-free rate
    :return: dataframe with NaNs filled with val
    """
    mask = ~(df.isna() & df.ffill().isna())
    mask &= df.isna()
    mask &= df.shift(1).notna()
    df[mask] = val
    return df

