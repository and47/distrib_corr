import pandas as pd

from input_proc import *  # xdf, ydf are company and market returns, respectively as pandas DataFrames
from p2_funs import *  # rolling_corr, fill_firstNaN_ingaps

# no significant discrepancies in the data, other than some missing values, maybe a couple at the
# beginning of the series, but nothing that would affect the correlation calculation

# in reality, we would want to check for outliers, but we'll skip that for now
# also, we would want to establish a starting point for the correlation calculation,
# as we don't want to include the initial NaNs in the calculation in case some stock was IPOed
# only years later the start of the market return series (e.g. 2000)

# it may be more efficient to calculate the pairwise rolling correlation
# using numpy, but pandas scales better (or at least easier with Dask) and PySpark
# also, it's possible to apply numpy functions to pandas DataFrame columns and rely on
# pandas to handle the indexing (rolling window) and alignment of the results (important for dates)
# One disadvantage of pandas is that doesn't preserve the memory layout of the original numpy array:
# regardless of whether original numpy array is in C (row-major) or Fortran (column-major) order,
# pandas DataFrame will be in C order, so it's possible to lose some performance (e.g. cache misses),
# which in practice I found to be ~3% on ~100MB data, unusually low impact

# Numpy has a function for rolling window, but it's cumbersome to use:
# np.lib.stride_tricks.sliding_window_view, and also the best algorithm would be
# online (incremental) calculation of the correlation such as Welford's algorithm,
# which would need to keep track of the mean and variance of the series,
# (rolling sum and sum of squares) and then calculate the correlation from that,
# so a 'rolling' behavior would need to be implemented manually as FIFO:
# by decrementing by first (oldest) observation and incrementing by the last (newest) these rolling sums
# code will be provided later, and if there's time, we can compare performance

# another advantage of Welford's algorithm is that it scales better in memory, and
# maybe outside of the scope of this task, but it can also allow to merge the results,
# (imagine it's like incrementing not by one observation, but by a chunk of observations).
# initially, we'd distributed dataset in Spark using smth like parquet files by columns,
# (each node having a market return series and a set of stock return series), but if necessary,
# as rows are also grouped into chunks, Spark can split them across the cluster too, and later,
# in theory, if there's need in Spark program to merge chunks of rows (e.g. from this parallelization),
# it will be technically possible

# for now in Pandas, we'll use these simple assumptions:
# - approximate 2-year rolling window, which is ~2x252, 505 business days, also may be a leap year
# - we'll use the default Pearson correlation
# - we'll use the default N-1 denominator for the sample correlation
# - we'll use the default min_periods=375, it's 75% of the window,
#    and at least 1.5ys of overlap, which is closer to 2ys than 1y
# - minor: neither of series has 0 variance over the period, so we won't get 0 in the denominator

# for local p2 task (Q2), we simply use pandas rolling corr function on each column of X DF against Y DF:
# it's easier to merge X and Y, so rolling can be done on single DF:
xydf = pd.concat([xdf, ydf], axis=1)
# xydf = pd.concat([xdf, ydf.to_frame('Y')], axis=1)  # in case need to name column uniquely


# see p2_funs.py
winsz = 505
minp = 375
xycorr = (
    xydf.
    rolling(window=winsz, min_periods=minp).
    corr(xydf.iloc[:, -1])  # pairwise rolling correlation of stock vs market returns
)  # in prod code, it's best to rely on names (unless for performance reasons) to reduce risk of bugs

xycorr.isna().mean().mean()  # 99% NaNs without any imputation
# to-do: try using time indexing (time windows size, actually 24 trailing months) instead of row indexing


# too many missing observations of companies, so need better NaN handling
# imputation can introduce bias, so will do it more carefully. forward filling one value
# doesn't do so (but only on price level), so with returns we need to impute 0 or risk-free rate,
# but in order for this not to affect result too much, we do it only for one observation (leave remaining continious NaNs)
# as we know that our data is randomly generated, we omit discussion of using mean, median or other advanced imputation


xydf = fill_firstNaN_ingaps(xydf, 0)  # reduces share of NaNs from 34% to 12%

xycorr = rolling_corr(xydf, winsz, minp)  # in prod code, it's best to rely on col names not idxs (unless for performance reasons) to reduce risk of bugs

xycorr.isna().mean().mean()  # 7% of NaNs, better than 99%, but still too much

# remove first 375 rows, as they are NaNs,
# also beware that last 125 rows are not based on full window,
# so we may also want exclude these, but this depends on the use case
xycorrd = xycorr.iloc[minp:]  # 4626 rows, 500-375+1 = 126 rows less than original 4752


# p4:
# performance can further be improved by using numpy Fortran arrays for cache locality, as
# we perform operations on columns. Also, as there are many companies, operation can be simply parallelized
# which can be done with Numba.  There are distributed algorithms for correlation, but they are
# not as useful here, where there's high number of columns and not so large window size, so parallelizing
# by rows is likely as unnecessary complication. Instead, we can parallelize by columns using GPU,
# by using CUDA, also supported by Numba, but that may be outside of the scope of this task or hardware limits
# extending Python with C++ can be even more performant than Numba, and there are various approaches: Cython, PyBind11, etc.
# we'd also need to consider what will work best with Spark UDFs, and also what will be easier to maintain

# in a perfect case, we'd keep track of performance of all these implementations using something like
# pytest-benchmark, (not only ensuring correctness with pytest), as "ranking" may change over time with
# newer version of Python, Numba, CUDA, Spark, etc.

import numpy as np

# Improvement -  implementation 1                    ##
# My vectorized NumPy (see contiguous/cache-locality and "SIMD" for why it improves performance)
#  implementation of Wikipedia "Online" (stable one-pass algorithm):
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
# Note: it's expanding window not rolling
def corr_np(a1: np.ndarray, a2: np.ndarray, retcov: bool = False, roll: bool = False) -> np.ndarray:
    # mean1 = mean2 = corr = N = 0
    assert a1.shape == a2.shape
    n = np.arange(1, a1.shape[0] + 1)
    csum = np.cumsum(np.stack([a1, a2], axis=0), axis=1)
    cmean = csum / n  # C-style contiguous, cache-locality, SIMD. so loop is still vectorized
    if roll:
        cov = np.full_like(n, fill_value=np.nan, dtype=float)
        for i in range(2, n[-1]):
            cov[i] = sum((a1[:i+1] - cmean[0, i]) * (a2[:i+1] - cmean[1, i])) / (n[i] - 1)
    else:
        cov = sum((a1 - cmean[0, -1]) * (a2 - cmean[-1, -1])) / (n[-1] - 1)
    if retcov:
        return cov
    return cov / np.sqrt(corr_np(a1, a1, retcov=True, roll=roll)*corr_np(a2, a2, retcov=True, roll=roll))

# this function can then be applied with Pandas apply on DF or on DF.values with np.apply_along_axis
# in Spark can potentially be tried with UDFs


# Improvement -  implementation 2                    ##
# Rolling window implemented in Numpy with efficient view, which should be faster than Pandas rolling
from typing import List
def corr_win_np(arrs: List[np.ndarray], winsz: int) -> np.ndarray:
    assert all(a.shape == arrs[0].shape for a in arrs)
    assert arrs[0].shape[0] >= winsz
    n = len(arrs)  # length
    vws = np.lib.stride_tricks.sliding_window_view(np.stack(arrs), (n, winsz))  # view, not copy
    return np.array(list(map(lambda x: np.corrcoef(x)[0,1], vws.squeeze())))  # apply along axis=1

# it's possible to implement also efficient algorithm, even in C:
# https://crypto.fit.cvut.cz/sites/default/files/publications/fulltexts/pearson.pdf
# and parallelize and optimize for AVX:
# https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf

# mathematically simpler can be approach based on totals:
# https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation/102982#102982

#from numba import njit  # optional further speed up, works also without numba (remove this and next line)
#@njit()
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


# Testing p4
a = np.random.random(size=8)
b = np.random.random(size=8)
window_size = 3

rolling_correlation(a, b, window_size)
rolling_correlation_numba(a, b, window_size)  # possible to parallelize with @njit(parrallel=True), see l.147
corr_win_np([a, b], window_size)              # by CPU cores or even GPU with CUDA for many variables a,b,c,d, etc.

