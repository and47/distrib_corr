import pandas as pd
import numpy as np
import os
from pathlib import Path

import datetime as dt

# loads common data, see tasks/Question specific .py and .ipynb files

xcret = pd.read_pickle('data/company_returns.pkl')
ymret = pd.read_pickle('data/market_returns.pkl')  # bdays only

xcret.loc[4999].sort_index().tail(10)  # also bdays only

# we know from how file was created that all the dates from stock returns are present in market
# returns, so we can just use the market returns as our main index

ymret.index.is_monotonic_increasing  # True
xcret.index.is_monotonic_increasing  # False, but will be sorted after unstack
xdf = xcret.unstack('companyid') # n x m arrays where n = number of dates, m = number of companies

assert xret.index.is_monotonic_increasing

xret = xret.asfreq(freq='B')  # change datetime freq to business days
xret.index

assert all(xret.index == ymret.index)

arr = np.asfortranarray(xret)  # columnar operation in theory faster on Fortran arrays,

# xdf is a pandas dataframe for company returns (X variables)
# ydf is a pandas dataframe for market returns (Y variable)

# xarr is a numpy F array for company returns (X variables)
# yarr is a numpy F array for market returns (Y variable)
