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
