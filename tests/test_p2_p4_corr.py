import os

import numpy as np
import pandas as pd

import numpy.ma as ma

import pytest

from p2_funs import rolling_corr, fill_firstNaN_ingaps
# data is imported from conftest.py

COL_LIMIT = 100  # limit number of columns for testing, as local Spark in p3 is slow and below will be compared to it

@pytest.fixture
def fixed_df(fixed_data) -> pd.DataFrame:
    mkt, cmpdf = fixed_data  # market and company returns DFs
    crets = cmpdf.unstack('companyid').asfreq(freq='B')
    rets = pd.concat([crets, mkt], axis=1)
    return rets

@pytest.fixture
def rndm_df(rndm_data) -> pd.DataFrame:
    mkt, cmpdf = rndm_data  # market and company returns DFs
    crets = cmpdf.unstack('companyid').asfreq(freq='B')
    rets = pd.concat([crets, mkt], axis=1)
    return rets

#@pytest.fixture  # separate fixture for random and fixed data?
def as_arr(df: pd.DataFrame) -> np.ndarray:
    return np.asfortranarray(df)

@pytest.fixture
def corr_params():
    return (505, 375)  # winsz: int, minp: int
@pytest.mark.parametrize('df', ['fixed_df', 'rndm_df'])
def test_corr_res(request, df, benchmark):
    rets = request.getfixturevalue(df)  # market and company returns
    rets = fill_firstNaN_ingaps(rets, 0)
    mcorr = rolling_corr(rets, winsz=10, minp=5)  # both pd and np.corr use N-1 in denominator
    # choose and check for some data point [100, 100] (arbitrary)
    p_marr = ma.array(rets.iloc[100:110, 100], mask=np.isnan(rets.iloc[100:110, 100]))
    npmacorr = ma.corrcoef(p_marr, rets.iloc[100:110, -1])  # numpy masked array
    assert np.allclose(mcorr.iloc[109, 100], npmacorr[0, 1], rtol=1e-4), "differs from numpy"


@pytest.mark.parametrize('df, maxcol', [['fixed_df', COL_LIMIT], ['rndm_df', COL_LIMIT]])
def test_corr_nans(request, benchmark, df, maxcol, corr_params):
    rets = request.getfixturevalue(df)  # market and company returns
    assert rets.index.is_monotonic_increasing
    rets = fill_firstNaN_ingaps(rets.drop(rets.columns[maxcol:-1], axis=1), 0)
    mcorr = benchmark(rolling_corr, rets, winsz=corr_params[0], minp=corr_params[1])
    nan_ratio = mcorr.isna().mean().mean()
    assert nan_ratio < 0.2, f"Too many NaNs in output, {nan_ratio:.2%}"
