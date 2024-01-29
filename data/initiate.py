import pandas as pd
import numpy as np
from itertools import repeat

def create_test_data():
    dates = pd.bdate_range('2000-01-01', '2023-05-12')
    N_companies = 5000
    N_dates = 4000
    index = [list(zip(repeat(i), np.random.choice(dates, size=N_dates, replace=False))) for i in range(N_companies)]
    index = [x for y in index for x in y]
    index = pd.MultiIndex.from_tuples(index, names=['companyid', 'date'])
    returns = np.random.normal(loc=0, scale=0.012, size=N_companies * N_dates)
    df = pd.Series(index=index, data=returns, name='returns')

    mkt_returns = np.random.normal(loc=0, scale=0.008, size=len(dates))
    market = pd.Series(index=dates, data=mkt_returns)
    
    df.to_pickle('company_returns.pkl')
    market.to_pickle('market_returns.pkl')

if __name__ == '__main__':
    create_test_data()

