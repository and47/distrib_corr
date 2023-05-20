import pandas as pd
import numpy as np
import os
from pathlib import Path


import datetime as dt

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


