import numpy as np
import pandas as pd
import pytest
from itertools import repeat

seed = 123

# TODAY = pd.Timestamp.today()
# CALSTART = dt.datetime(2000, 1, 1)
# S_YR2K = '2000-01-01'  # some globals may be useful in future, as already do repeat in below code

genor_ran = np.random.default_rng()  # new per NEP 19, can be used with Cython
genor_fix = np.random.default_rng(seed)

# initiate one fixed fixture with preset seed (for reproducible results, benchmarking)
# and one with random values (for testing universal application of functions)

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
    market, df = create_test_data(genor_fix, dates=pd.bdate_range('2000-01-01', '2023-05-20'))
    return market, df

@pytest.fixture
def rndm_data() -> tuple:
    market, df = create_test_data(genor_ran)
    return market, df
