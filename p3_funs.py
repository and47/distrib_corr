import logging

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window


def hdist_roll_corr(df: DataFrame, window: int = 500) -> DataFrame:
    """Calculate the rolling correlation of 'X' with 'market'.

    Parameters:
    :param df: pyspark.sql.dataFrame.DataFrame data in columns ['timestamp', 'market', 'X', 'value']
        representing over time returns of the market and, in a long format, stock 'X' in value column
    :window: the rolling window size as a number of rows (pairs of observations of X and market)
    :return: the output DataFrame with added 'rolling_corr' column
    """

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        # logging.info("Starting calculation...")  # this is done lazily

        # Repartition on the 'X' column, can allow to parameterize the number of chunks
        # df = df.repartition(min(nchunks, int(df.select(F.countDistinct('X')).toPandas().values)), 'X')
        df = df.repartition('X')

        # Define the window specification
        windowSpec = Window.partitionBy('X').orderBy(F.col('timestamp')).rowsBetween(-(window-1), 0)

        # Calculate the rolling correlation
        df = df.withColumn('rolling_corr', F.corr(F.col('value'), F.col('market')).over(windowSpec))

        # logging.info("Calculation completed successfully.")  # this is done lazily, so it's printed too early

        return df

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None
