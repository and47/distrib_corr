import pandas as pd
import numpy as np
import datetime as dt

def float_to_uint_minret(arr: np.ndarray, prec: int = 4, dtype_: np.dtype = np.uint16, minval: float = -1)\
        -> np.ndarray:
    """
    Convert float64 array to integers of specified type using given levels of minval and precision.

    To save space by rounding and storing as integers e.g. stock returns represented as offsets
     (of steps sized according to level of precision) from minimum possible return (e.g. -1 or -100%).
     E.g. uint16 array with minimum value of minval and precision of prec digits after the decimal point.

    :param arr: array of float64 values representing stock returns
    :param prec: number of digits after the decimal point to keep
    :param dtype_: integer data type to use for storing the result (should be unsigned and have enough bits for prec)
    :param minval: minimum possible value for stock return (e.g. -1 or -100%), used to calculate offset
    :return:
    """
    assert arr.ndim == 2
    assert arr.dtype == np.float64
    stepsz = 10**(-prec)  # step size for given precision; reserve 0 for nan, (first +stepsz) for actual minval:
    rarr = arr + (-minval + stepsz) # can also use arr.min(), but solving for general case (not random data)
    assert np.nanmin(rarr) > 0, f"unexpected value for stock return < minval ({minval})"
    # rarr = rarr.round(prec) / stepsz  # alternatively, we can use rarr // stepsz
    rarr = rarr.round(prec) * 10**prec  # alternatively, we can use rarr // stepsz
    assert np.nanmax(rarr) <= np.iinfo(dtype_).max, f"overflown {dtype_} with > {np.iinfo(dtype_).max}"
    rarr = rarr.astype(dtype_) # reserve 0 for nan. 1 is -1 (lowest possible value for stock return)
    return rarr

def persist_rets_bin(fname: str, arr: np.ndarray | pd.DataFrame,
                     calstart: np.datetime64 = dt.datetime(2000, 1, 1),
                     cat_dtype: np.dtype = np.uint16,
                     dat_dtype: np.dtype = np.uint16,
                     to_int: np.dtype = None, **kwargs) -> None:
    """
    Persist returns to binary file with numpy.savez_compressed.

    Optionally, save metadata such as time index and company ids
     and convert to integer values.

    :param fname: file name
    :param arr: returns array or DataFrame with time index and company ids
     in columns
    :param calstart: calendar start date, used to calculate business day offsets
    :param cat_dtype: company id data type (depends on number of companies)
    :param dat_dtype: date offset data type (depends on number of dates since calstart)
    :param to_int: convert returns to integer values (e.g. for uint16)
    :param kwargs: extra arguments (precision, minreturn, or maxreturn),
     see floatr_to_uint_minret
    """
    assert arr.ndim == 2
    firms = bd_offs = None
    if isinstance(arr, pd.DataFrame):
        assert arr.columns.max() < np.iinfo(cat_dtype).max, \
            f"overflown companyid {cat_dtype} with > {np.iinfo(cat_dtype).max}"
        firms = arr.columns.to_numpy(dtype=cat_dtype)
        sdate = calstart.date()  # get business day count (offset) from this date, we know that 2000 is earliest year
        assert np.busday_count(sdate, arr.index.date.max()) < np.iinfo(dat_dtype).max,\
            f"overflown dates delta {dat_dtype} with > {np.iinfo(dat_dtype).max}"
        bd_offs = 1 + np.array([np.busday_count(sdate, x) for x in arr.index.date], dtype=dat_dtype)
        arr = arr.to_numpy()  # this used to allow to specify F or C order, but now it's always C
    arr = np.asfortranarray(arr)
    # assert arr.dtype == np.float64  # now also using to save example offset int values:
    if to_int:
        arr = float_to_uint_minret(arr, dtype_=to_int, **kwargs)
    np.savez_compressed(str(fname) + '_npcomprsd', arr=arr, firms=firms, bd_offs=bd_offs)



def uint_ret_offs_tofloat(arr: np.ndarray, prec=4, minval=-1, dtype_: np.dtype = np.uint16) -> np.ndarray:
    """
    Convert integer array to float64 array using given levels of minval and precision.

    To restore float64 array from integers representing stock returns as offsets
     (of steps, sized according to level of precision) from minimum possible return (e.g. -1 or -100%).
     E.g. uint16 array with minimum value of minval and precision of prec digits after the decimal point.

    :param arr: array of integers representing stock returns
    :param prec: number of digits after the decimal point that was kept
    :param dtype_: integer data type used for storing the result
    :param minval: minimum possible value for stock return (e.g. -1 or -100%), which was used to calculate offset
    :return: array of float64 values representing stock returns
    """
    assert arr.ndim == 2
    assert arr.dtype == dtype_
    out = np.full_like(arr, np.nan, dtype=np.float64)
    stepsz = 10**(-prec)
    out[arr != 0] = arr[arr != 0] * stepsz - (-minval + stepsz)
    return out


def load_rets_from_bin(fpath: str, as_numpy: bool = False,
                       calstart: np.datetime64 = dt.datetime(2000, 1, 1),
                       cat_dtype: np.dtype = np.uint16,
                       dat_dtype: np.dtype = np.uint16,
                       val_dtype: np.dtype = None, prec: int = None,
                       minval: float = None, maxval: float = None) -> pd.DataFrame | np.ndarray:
    """
    Load stock returns from binary file into DataFrame or ndarray.

    Reconstructs return values and metadata (columns with company ids and index of dates).

    :param fpath: path to binary file to load data from
    :param as_numpy: True to return ndarray, False to return DataFrame
    :param calstart: calendar start date, which was used to calculate business day offsets
    :param cat_dtype: company id data type used (usually depends on number of companies)
    :param dat_dtype: date offset data type used (usually depends on number of dates since calstart)
    :param val_dtype: integer data type if used for storing the result as offets from minval
    :param prec: number of digits after the decimal point that was kept [to-do: save this as metadata to auto-reconstruct]
    :param minval: minimum possible value for stock return (e.g. -1 or -100%), used to calculate offset
    :return: a table of stock returns (2D ndarray or Pandas DataFrame)
    """
    uncompressed_arrs = np.load(file=fpath)
    arr = uncompressed_arrs['arr']
    if val_dtype is None or val_dtype == np.float64:
        val_dtype = np.float64
        assert arr.dtype == val_dtype
    else:
        arr = uint_ret_offs_tofloat(arr, prec=prec, minval=minval, dtype_=val_dtype)
    if as_numpy:
        return arr
    else:
        assert arr.ndim == 2, "Pandas DataFrame can only be constructed from 2D array, a table expected"
        firms = uncompressed_arrs['firms']
        from pandas.tseries.offsets import BDay  # move to top if will be used elsewhere
        bd_offs = uncompressed_arrs['bd_offs']
        assert arr.shape == (bd_offs.shape[0], firms.shape[0])
        assert firms.dtype == cat_dtype
        assert bd_offs.dtype == dat_dtype
        dates = np.array([(calstart + BDay(x)).date() for x in bd_offs])
        return pd.DataFrame(arr, index=dates, columns=firms)
