import pandas as pd
import numpy as np
import pickle
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from sts_backend.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # # Compress raw_data by setting types to DTYPES_RAW
    # df = df.astype(DTYPES_RAW)

    # # Remove buggy transactions
    # df = df.drop_duplicates()  # TODO: handle whether data is consumed in chunks directly in the data source
    # df = df.dropna(how='any', axis=0)

    # df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
    #                 (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]

    # df = df[df.passenger_count > 0]
    # df = df[df.fare_amount > 0]

    # # Remove geographically irrelevant transactions (rows)
    # df = df[df.fare_amount < 400]
    # df = df[df.passenger_count < 8]

    # df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    # df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    # df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    # df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

    # print("✅ data cleaned")

    return df

def get_data_with_cache(cache_path:Path) -> np.ndarray:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local pickle..." + Style.RESET_ALL)
        """new code"""
        with open(cache_path, 'rb') as file:
            data = pickle.load(file)
        print(f"✅ Data loaded, with shape {data.shape}")
        return data
    else:
        print("There is no file.")
        return None
