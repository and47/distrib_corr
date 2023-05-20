import pandas as pd
import numpy as np
import os
from pathlib import Path


import datetime as dt

from p1_funs import *

xcret = pd.read_pickle('data/company_returns.pkl')
ymret = pd.read_pickle('data/market_returns.pkl')  # bdays only

xcret.loc[4999].sort_index().tail(10)  # also bdays only

# we know from how file was created that all the dates from stock returns are present in market
# returns, so we can just use the market returns as our main index

ymret.index.is_monotonic_increasing  # True
xcret.index.is_monotonic_increasing  # False, but will be sorted after unstack
xret = xcret.unstack('companyid') # n x m arrays where n = number of dates, m = number of companies


assert xret.index.is_monotonic_increasing

xret = xret.asfreq(freq='B')  # change datetime freq to business days


assert all(xret.index == ymret.index)


# quick search of compressors suggested to use blosc2
import blosc2 as bl2
blarr = bl2.compress2(np.ascontiguousarray(arr))

savedsz = bl2.save_array(blarr, str(Path(fname, 'gfdgdf.bl2')), mode='w')
# worse than npz

import h5py
with h5py.File(str(Path(fname, 'cr_compressed9.h5')), "w") as f:
    # same size
    # dset_coefs = f.create_dataset("a", data=arr, compression="gzip", compression_opts=9)
    dset_coefs = f.create_dataset("a", data=arr, compression="gzip", compression_opts=9, shuffle=True)
    # dset_coefs = f.create_dataset("a", data=arr, compression="SZIP", compression_opts=9)
    # dset_coefs = f.create_dataset("a", data=arr, compression="ZFP", compression_opts=9)
    # dset_coefs = f.create_dataset("a", data=arr, compression="LZF", compression_opts=9)
    # dset_coefs = f.create_dataset("a", data=arr, compression="LZMA", compression_opts=9)


import pyarrow as pa
import pyarrow.parquet as pq

# Convert pandas DataFrame to PyArrow Table
table = pa.Table.from_pandas(xret)

# Write PyArrow Table to Parquet file
pq.write_table(table, 'array.parquet')  # already less than pkl only 20% more than np compressed
pq.write_table(table, 'xretgzipped.parquet', compression='gzip')  # already less than pkl only 20% more than np compressed
# gzip only 3% improvement over uncompressed parquet, zstd same, no improvement with brotli either


# it's possible to decrease further with better compression tools or
# by losing precision (float32) or by using a different format (e.g. hdf5, shuffling here won't help)
# this decreases size of nobs * ncomps * 8 bytes for float64 from 232 to 160 MB

# also, we can use min and max values from data to create custom dtype:
np.nanmax(arr)
np.nanmin(arr)

# but this is a bit like cheating, instead we can make use of the fact that stock returns are > -1


# numpy.uint16  # 0 to 65535, whereas uint8 is 0 to 255 (not enough for either company ids or dates)


# 32-bit floating-point values cover range from 1.175494351 * 10^-38 to 3.40282347 * 10^+38,
# whcih is too much for stock returns, so we may want to limit precision to 4 or 5 digits after the decimal point
# (e.g. 1.2345e-2) and use integers to count number of steps (e.g. 0.0001 step size for 4 digits after the decimal point)
# from minimal value, which for stock returns is -1, so we can use uint16 for that:

# first, use defined function to convert float64 to uint16 per this logic:


a = float_to_uint_minret(xret.values, prec=2)
a = float_to_uint_minret(xret.values, prec=5, minval=-0.1)  # can use for our random data
a = float_to_uint_minret(xret.values, prec=4, minval=-1)  # appropriate defaults for stock returns

np.unique(a)
a[a>0].min()
a.max()

arr= xret
calstart = dt.datetime(2000, 1, 1)

fname_ = Path(os.getcwd())

persist_rets_bin(str(fname_) + r'\fl', xret)
persist_rets_bin(str(fname_) + r'\uint16off', xret, to_int=np.uint16)
# 80% reduction in size, but obviously we lose precision, may be outside of the task scope


xret2 = load_rets_from_bin('uint16off_npcomprsd.npz', val_dtype=np.uint16, prec=4, minval=-1)

prec = 4
stepsz = 10**(-prec)  # fails with smaller rounding or np.close 0
assert np.nanmax(np.abs(xret.values.round(4) - xret2.values)) < stepsz, "reconstructed values differ from original"
# may still some bug in my code, which minimally distorts precision, needs improvement before production use

