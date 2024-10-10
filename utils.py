from typing import Iterable
import pandas as pd
import logging

# Given a dataframe and a set of indices, split the dataframe into 2 dataframes
# The first has all rows with a index within the set, and the second has all rows with an index not in the set
def split_df(df: pd.DataFrame, indices: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df.index.isin(indices)], df[~df.index.isin(indices)]

def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():  # To prevent adding handlers multiple times
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def feature_label(feature_subset: Iterable[str]):
    return "_".join(sorted(list(feature_subset)))
