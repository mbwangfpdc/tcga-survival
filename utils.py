from typing import Iterable
from itertools import chain, combinations
from globals import INCOMPATIBLE_FEATURE_TYPES, FEATURE_ID_DELIM
import pandas as pd
import logging
import datetime


# Given a dataframe and a set of indices, split the dataframe into 2 dataframes
# The left DF is all rows not in the inputted set, and the right is all rows specified by the set
def split_df(df: pd.DataFrame, indices: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[~df.index.isin(indices)], df[df.index.isin(indices)]


# Get a timestamp, which could be used for identifying the results of a run
def get_timestamp() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")


def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():  # To prevent adding handlers multiple times
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


# Given the types of features, return the label
def feature_id(feature_subset: Iterable[str]):
    return FEATURE_ID_DELIM.join(sorted(list(feature_subset)))


def model_type_and_fold(model_id: str) -> tuple[str, int]:
    split = model_id.rsplit("_", 1)
    return split[0], int(split[1])


# Given a set, return a set containing all possible subsets
# We return frozensets as they are hashable
def powerset(universe: Iterable) -> set[frozenset]:
    # Credit to https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    return set(
        [
            frozenset(combo)
            for combo in chain.from_iterable(
                combinations(universe, r) for r in range(len(universe) + 1)
            )
        ]
    )


def get_feature_sets(features: Iterable[str]) -> list[set]:
    pset = powerset(features)
    pset.remove(frozenset())
    for subset in list(pset):
        # Remove any generated subsets with incompatible feature types
        for ift_set in INCOMPATIBLE_FEATURE_TYPES:
            if ift_set.issubset(subset):
                pset.remove(subset)
                break
    return list(pset)


# Return a list of all pairs of subsets which combine to make the input set, not including the empty set
# For example, set_partitions({1, 2, 3}) would return [({1}, {2,3}), ({2}, {1,3}), ({3}, {1,2})]
def set_partitions(s: set) -> list[tuple[set, set]]:
    pset = powerset(s)
    pset.remove(frozenset())
    pset.remove(frozenset(s))
    returned = set()
    res = []
    for subset in pset:
        if subset in returned:
            continue
        complement = s.difference(subset)
        returned.add(subset)
        returned.add(complement)
        res.append((subset, complement))
    return res
