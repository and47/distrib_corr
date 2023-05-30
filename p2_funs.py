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



# for p4 (faster versions of p2):
from typing import List
import numpy as np


def corr_win_np(arrs: List[np.ndarray], winsz: int) -> np.ndarray:
    assert all(a.shape == arrs[0].shape for a in arrs)
    assert arrs[0].shape[0] >= winsz
    n = len(arrs)  # length
    vws = np.lib.stride_tricks.sliding_window_view(np.stack(arrs), (n, winsz))  # view, not copy
    return np.array(list(map(lambda x: np.corrcoef(x)[0,1], vws.squeeze())))  # apply along axis=1


def rolling_correlation(a, b, window_size):
    assert len(a) == len(b)
    assert len(a) >= window_size

    # Create empty array to hold correlation coefficients
    r = np.empty(len(a) - window_size + 1)

    # Calculate initial window sums
    sum_x = np.sum(a[:window_size])
    sum_y = np.sum(b[:window_size])
    sum_xy = np.sum(a[:window_size] * b[:window_size])
    sum_x2 = np.sum(a[:window_size] ** 2)
    sum_y2 = np.sum(b[:window_size] ** 2)

    for i in range(len(r)):  # not possible to parallelize with numba (due to exchange of data between iterations
        if i > 0:  # If not the first window, update sums
            sum_x = sum_x - a[i - 1] + a[i + window_size - 1]
            sum_y = sum_y - b[i - 1] + b[i + window_size - 1]
            sum_xy = sum_xy - a[i - 1] * b[i - 1] + a[i + window_size - 1] * b[i + window_size - 1]
            sum_x2 = sum_x2 - a[i - 1] ** 2 + a[i + window_size - 1] ** 2
            sum_y2 = sum_y2 - b[i - 1] ** 2 + b[i + window_size - 1] ** 2

        numerator = window_size * sum_xy - sum_x * sum_y
        denominator = np.sqrt((window_size * sum_x2 - sum_x ** 2) * (window_size * sum_y2 - sum_y ** 2))

        r[i] = numerator / denominator

    return r

from numba import njit
@njit()
def rolling_correlation_numba(a, b, window_size):
    assert len(a) == len(b)
    assert len(a) >= window_size

    # Create empty array to hold correlation coefficients
    r = np.empty(len(a) - window_size + 1)

    # Calculate initial window sums
    sum_x = np.sum(a[:window_size])
    sum_y = np.sum(b[:window_size])
    sum_xy = np.sum(a[:window_size] * b[:window_size])
    sum_x2 = np.sum(a[:window_size] ** 2)
    sum_y2 = np.sum(b[:window_size] ** 2)

    for i in range(len(r)):  # not possible to parallelize with numba (due to exchange of data between iterations
        if i > 0:  # If not the first window, update sums
            sum_x = sum_x - a[i - 1] + a[i + window_size - 1]
            sum_y = sum_y - b[i - 1] + b[i + window_size - 1]
            sum_xy = sum_xy - a[i - 1] * b[i - 1] + a[i + window_size - 1] * b[i + window_size - 1]
            sum_x2 = sum_x2 - a[i - 1] ** 2 + a[i + window_size - 1] ** 2
            sum_y2 = sum_y2 - b[i - 1] ** 2 + b[i + window_size - 1] ** 2

        numerator = window_size * sum_xy - sum_x * sum_y
        denominator = np.sqrt((window_size * sum_x2 - sum_x ** 2) * (window_size * sum_y2 - sum_y ** 2))

        r[i] = numerator / denominator

    return r