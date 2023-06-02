import pandas as pd
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

import pyarrow as pa
import pyarrow.parquet as pq

from p2_funs import fill_firstNaN_ingaps
from p3_funs import hdist_roll_corr
# from p2_corr import winsz, minp  # add __main__ to p2_corr.py to not run it on import
winsz = 505  # temporary workaround

from input_proc import *  # xdf, ydf are company and market returns, respectively as pandas DataFrames

# Convert pandas DataFrame to PyArrow Table; in practice, we can read PyArrow from parquet file directly,
# but in the exercise we manually create file for PySpark to read

ydf.name = 'market'  # name column of market returns
xdf = fill_firstNaN_ingaps(xdf, 0)  # fill first NaNs in gaps of company return data

xydf = pd.concat([xdf, ydf], axis=1)

xydfl = (xydf
          .melt(ignore_index=False, id_vars=['market'], value_name='value', var_name='X')
          .reset_index(names='timestamp')
         )

longtbl = pa.Table.from_pandas(xydfl)  # prepare data in long format for PySpark
pq.write_table(longtbl, 'lxy.parquet')  # best to store it sorted by X and by timestamp, but will be done by Spark


# One disadvantage of pandas mentioned earlier that it doesn't preserve the memory layout of the original numpy array:
# regardless of whether original numpy array is in C (row-major) or Fortran (column-major) order, is no
# longer a problem with PyArrow, as it's designed to be a columnar format, and should be more efficient as F-order
# The issue is more important for wide data, which can be worked with columnar format in manually set up MapReduce
# jobs, but in current example, Spark abstracts many such optimizations away

# spark = SparkSession.builder.appName("app").getOrCreate()  # for use in cluster mode. Below for local mode:
os.environ['PYSPARK_PYTHON'] = 'C:/Users/admin/.conda/envs/main/python.exe'  # required but may be moved to conda yaml
# os.environ["PYSPARK_DRIVER_PYTHON_OPTS"] = "-Xms4g -Xmx4g"  # may need to set the driver memory
spark = (SparkSession.builder
            .appName("app")
            .master('local[*]')
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate()
         )

df = spark.read.parquet("lxy.parquet")  # same as longtbl, load the dataframe as if in a batch job on a cluster

w_corr = hdist_roll_corr(df, winsz)  # see p3_funs.py for implementation

w_corr.explain()  # execution plan

# res = w_corr.toPandas()  # too big on full data, will error with 4gb driver memory
res = w_corr.filter(w_corr.X < 10).toPandas()  # first 10 companies
# res = w_corr.filter(w_corr.X.isin([1, 99, 4998])).toPandas()  # or select specific ones






# Alternative solutions with Wide input data
# pq.write_table(xydf, 'wxy.parquet')  # in practice, we can read PyArrow from parquet file directly, but in the exercise we manually create file for PySpark to read
# df = spark.read.parquet("wxy.parquet")  # wide data (x firms, y market)

# 1. Use pandas_udf to apply rolling_corr to each column (see 2. doesn't work):
# selected_columns = df.columns[-10:]
# df_selected = df.select(selected_columns)
#
# new_column_names = ['x' + i if i not in ['market', '__index_level_0__'] else i for i in df.columns]  # columns starting with digit not allowed in spark
# df = df.toDF(*new_column_names)
#
# # Convert target column to pandas series and broadcast
# target_column = df.select('market').toPandas().squeeze()
# broadcast_target = spark.sparkContext.broadcast(target_column)
#
# # Define pandas udf function
# def rolling_corr(column: pd.Series) -> pd.Series:
#     target_column = broadcast_target.value
#     return column.rolling(window=winsz, min_periods=minp).corr(target_column)
#
# rolling_corr_udf = F.pandas_udf(rolling_corr, T.DoubleType())
#
# # new_df = spark.createDataFrame([], schema=df.schema)  # this will lead to warning of adding columns with same name
# new_df = spark.createDataFrame([], schema=T.StructType([]))  # errors: pyspark.sql.utils.AnalysisException: Resolved attribute(s) x0#35018 missing from  in operator !Project [rolling_corr(x0#35018) AS x0#45024].;
# new_df = spark.createDataFrame([], schema=T.StructType([T.StructField(col_name, T.DoubleType(), nullable=True) for col_name in new_column_names]))
#
# # Apply the function to each column and write result to new dataframe
# for col in df.columns:
#     if (col != 'market') and (col != '__index_level_0__'):
#         new_df = new_df.withColumn(col, rolling_corr_udf(df[col]))
#
# new_df.explain()  # local: check execution plan
# new_df.show()

# 2. Use Window (works on smaller dataset, 10 below), idea is to map columns (companies) to nodes
#  and reduce (aggregate) all correlation on driver node.  However, this needs very careful control of
#  partitioning and shuffling, which PySpark doesn't seem to easily allow for columns
#
#
# df_selected = df.select(df.columns[-10:])  # see 1. for reading file and creating df
# new_column_names = ['x' + i if i not in ['market', '__index_level_0__'] else i for i in df_selected.columns]  # columns starting with digit not allowed in spark
# df_selected = df_selected.toDF(*new_column_names)
#
# windowSpec = Window.orderBy(F.col('__index_level_0__')).rowsBetween(-9, 0)  # Define the window as the last 10 rows
# column_names = df_selected.columns[:-2]
#
# dfs = []
#
# for col_name in column_names:
#     # Calculate the rolling correlation
#     df_corr = df_selected.withColumn(f'{col_name}_corr', F.corr(F.col(col_name), F.col('market')).over(windowSpec))
#     # Append the DataFrame to the list
#     dfs.append(df_corr.select(f'{col_name}_corr', "__index_level_0__"))
#
# df_final = dfs[0]
# for df in dfs[1:]:
#     df_final = df_final.join(df, on="__index_level_0__", how="inner")
#
# df_final.explain()
# finalsee = df_final.toPandas()


# to-do to try:
#  broadcasting cols market and time
#  caching them
#  using custom UDF or numpy fun from p2_funs.py instead of Pyspark window
#  more advanced window specs, taking into account business days/calendar or missing values
#  various cluster optimizations (settings, configs, etc)
