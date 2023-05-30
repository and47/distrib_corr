import os

import numpy as np
import pandas as pd

import numpy.ma as ma

import pytest

from p2_funs import rolling_corr  #, fill_firstNaN_ingaps, for test purposes filling all NaNs with 0

os.environ['PYSPARK_PYTHON'] = 'C:/Users/admin/.conda/envs/main/python.exe'
# data is imported from conftest.py
pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_SLOW_TESTS") and os.environ.get("PYSPARK_PYTHON")),
    reason="PYSPARK_PYTHON env var required. Also, this test is slow or requires additional dependencies. Not ran by default. Set RUN_SLOW_TESTS=1 to run."
)

from p3_funs import hdist_roll_corr

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


# data in long and wide format, with some NaNs filled in (see p2_funs.py)
def readyLongWideDFs(data: tuple, limit: int = None) -> tuple:
    mkt, cmpdf = data  # market return Series and company returns DF
    mkt.name = 'market'  # name column of market returns Series
    if limit:
        cmpdf = cmpdf[cmpdf.index.get_level_values('companyid') < limit]
    wcrets = (cmpdf.unstack('companyid')
              .asfreq(freq='B')
              .fillna(0))  # wide
    wdf = pd.concat([wcrets, mkt], axis=1)
    ldf = (wdf
           .melt(ignore_index=False, id_vars=['market'], value_name='value', var_name='X')
           .reset_index(names='timestamp'))  # long
    return ldf, wdf

@pytest.fixture
def fixed_lwdfs(fixed_data) -> pd.DataFrame:
    return readyLongWideDFs(fixed_data, COL_LIMIT)

@pytest.fixture
def rndm_lwdfs(rndm_data) -> pd.DataFrame:
    return readyLongWideDFs(rndm_data, COL_LIMIT)

from test_p2_p4_corr import corr_params, COL_LIMIT  # test fixture import

@pytest.fixture(scope='module')
def spark():
    spark = (SparkSession.builder
             .appName("app")
             .master('local[*]')
             .config("spark.driver.memory", "4g")
             .config("spark.executor.memory", "4g")
             .getOrCreate()
             )
    yield spark
    spark.stop()

# for simplicity only use fixed data, avoid request fixture
def test_corr_res(fixed_lwdfs, corr_params, spark, benchmark):
    pdcorr = rolling_corr(fixed_lwdfs[1], winsz=corr_params[0], minp=1)  # both pd and PySpark use N-1 in denominator
    # choose and check for some company e.g. 5 (arbitrary column):
    ldf = spark.createDataFrame(fixed_lwdfs[0])
    spcorr = benchmark.pedantic(hdist_roll_corr, kwargs={'df': ldf, 'window': corr_params[0]},
                                rounds=3)  # spark, limit number of rounds for benchmarking (usually over 100)
    # spcorr = hdist_roll_corr(ldf, corr_params[0])  # or just run once without benchmarking to save time
    spcorr5 = spcorr.filter(spcorr.X == 5).toPandas()['rolling_corr']
    pdcorr5 = pdcorr.loc[:, 5].values
    assert spcorr5.values.shape == pdcorr5.shape, "spark p3 corr shape differs from pd p2 implementation"
    assert np.allclose(spcorr5.values, pdcorr5, rtol=1e-4, equal_nan=True),\
        "spark p3 corr differs from pd p2 implementation"
