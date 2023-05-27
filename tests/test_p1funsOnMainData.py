import numpy as np
import pandas as pd

import pytest
import tempfile  # use to test compression

#import main
from p1_funs import persist_rets_bin, load_rets_from_bin, floatr_to_uint_minret, uint_ret_offs_tofloatr

# in conftest.py, we have:
#  initiated one fixed fixture with preset seed (for reproducible results, benchmarking)
#  and one with random values (for testing universal application of functions)


# unfortunately, benchmark plugin only tests speed, not memory (disk usage); so not for compression comparison
#  so this wasn't included in tests also in the interest of time

def test_initiate(fixed_data, rndm_data):
    market, df = fixed_data
    xret = df.unstack('companyid')
    assert len(market.shape) == 1 and market.shape[0] == xret.shape[0]
    assert market.index.is_monotonic_increasing
    # assert xret.shape == (6095, 5000)
    exp_rows = np.busday_count(pd.to_datetime('2000-01-01').date(), pd.Timestamp.today().date())
    assert xret.shape == (6100, 5000)  # for random changes with run day (uses today's date)
    assert ~np.array_equal(rndm_data[0], market)  # different but equivalent shape
    assert rndm_data[0].shape == exp_rows, market.shape[1]
    dfran = rndm_data[1].unstack('companyid')
    assert ~np.array_equal(dfran, xret)
    assert dfran.shape == (exp_rows, xret.shape[1])
    print('random data (mkt and stock mins): ', np.nanmin(rndm_data[0]), np.nanmin(dfran))  # should be different
    print(' fixed data (mkt and stock mins): ', np.nanmin(market), np.nanmin(df))  # should be:
    # -0.028345506444168257 -0.06909354203029808

@pytest.mark.parametrize('data', ['fixed_data', 'rndm_data'])
def test_conversion(request, data: tuple, benchmark):
    _, df = request.getfixturevalue(data)  # market not used
    crets = df.unstack('companyid')
    assert crets.index.is_monotonic_increasing
    crets = crets.to_numpy()
    crets_rc = floatr_to_uint_minret(crets, prec=4, minval=-1)  # rounded and converted
    assert crets_rc.shape == crets.shape
    assert crets_rc.dtype == np.uint16
    crets_re = uint_ret_offs_tofloatr(crets_rc, prec=4, minval=-1)  # restored/reconstructed from uint16 object
    assert np.nanmin(crets_re) >= -1
    assert crets_re.shape == crets.shape
    assert crets_re.dtype == crets.dtype
    assert np.nanmax(np.abs(crets_re - crets)) < 10**(-4), "reconstructed values differ from original"
    assert np.allclose(np.nan_to_num(crets_re), np.nan_to_num(crets), atol=1e-4), "reconstructed values differ from original"  # 1e fails

@pytest.mark.parametrize('data', ['fixed_data', 'rndm_data'])
def test_overflow(request, data: tuple):
    _, df = request.getfixturevalue(data)  # market not used
    with pytest.raises(AssertionError):  # to do: use Exceptions in actual code, leave asserts for testing (test_* files)
        crets_re = floatr_to_uint_minret(df.values, prec=6, minval=-1)

#
# test_reconstruction
#
# test_savedsize


