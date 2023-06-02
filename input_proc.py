import pandas as pd
import numpy as np
import os
from pathlib import Path

import datetime as dt

# loads common data, see tasks/Question specific .py and .ipynb files
fname = Path(os.getcwd())
if not Path.exists(Path('data/company_returns.pkl')):
    import subprocess
    subprocess.run(['python', 'initiate.py'], cwd='./data')
xcret = pd.read_pickle('data/company_returns.pkl')
ydf = pd.read_pickle('data/market_returns.pkl')  # bdays only, doesn't require processing, hence naming

xcret.loc[4999].sort_index().tail(10)  # bdays only, but need to set freq to ensure pd uses index correctly in ops

# we know from how file was created that all the dates from stock returns are present in market
# returns, so we can just use the market returns as our main index

ydf.index.is_monotonic_increasing  # True
xcret.index.is_monotonic_increasing  # False, but will be sorted after unstack
xdf = xcret.unstack('companyid')  # n x m arrays where n = number of dates, m = number of companies

assert xdf.index.is_monotonic_increasing

xdf = xdf.asfreq(freq='B')  # change datetime freq to business days
xdf.index

assert all(xdf.index == ydf.index)

xarr = np.asfortranarray(xdf)  # columnar operation in theory faster on Fortran arrays,
yarr = np.asfortranarray(ydf)  # in practice, so far seen 3% speedup even on ~100MB data

# xdf is a pandas dataframe for company returns (X variables)
# ydf is a pandas dataframe for market returns (Y variable)

# xarr is a numpy F array for company returns (X variables)
# yarr is a numpy F array for market returns (Y variable)
