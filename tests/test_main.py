import numpy as np
import pandas as pd

import pytest
#import main

# initiate one fixed fixture with preset seed (for reproducible results, benchmarking)
# and one with random values (for testing universal application of functions)

from itertools import repeat

seed = 123

genor_ran = np.random.default_rng()  # new per NEP 19, can be used with Cython
genor_fix = np.random.default_rng(seed)


def create_test_data(genor, N_companies = 5000, N_dates = 4000, dates = pd.bdate_range('2000-01-01', pd.Timestamp.today())):
    index = [list(zip(repeat(i), genor.choice(dates, size=N_dates, replace=False))) for i in range(N_companies)]
    index = [x for y in index for x in y]
    index = pd.MultiIndex.from_tuples(index, names=['companyid', 'date'])
    returns = genor.normal(loc=0, scale=0.012, size=N_companies * N_dates)
    df = pd.Series(index=index, data=returns, name='returns')
    mkt_returns = genor.normal(loc=0, scale=0.008, size=len(dates))
    market = pd.Series(index=dates, data=mkt_returns)
    return market, df

# see seed, even fixed not the same as currently used in main (to-do?)
@pytest.fixture
def fixed_data() -> tuple:
    market, df = create_test_data(genor_fix)
    return market, df

@pytest.fixture
def rndm_data() -> tuple:
    market, df = create_test_data(genor_ran)
    return market, df

def test_initiate(fixed_data, rndm_data):
    market, df = fixed_data
    xret = df.unstack('companyid')
    assert len(market.shape) == 1 and market.shape[0] == xret.shape[0]
    assert market.index.is_monotonic_increasing
    # assert xret.shape == (6095, 5000)  # changes with run day (uses today's date)
    assert xret.shape
    assert ~np.array_equal(rndm_data[0], market)  # different but equivalent shape
    assert rndm_data[0].shape == market.shape
    dfran = rndm_data[1].unstack('companyid')
    assert ~np.array_equal(dfran, xret)
    assert dfran.shape == xret.shape
    print('random data (mkt and stock mins): ', np.nanmin(rndm_data[0]), np.nanmin(dfran))  # should be different
    print(' fixed data (mkt and stock mins): ', np.nanmin(market), np.nanmin(df))  # should be:
    # -0.028345506444168257 -0.06909354203029808
#
# test_overflow
#
# test_coversion
#
# test_reconstruction
#
# test_savedsize
