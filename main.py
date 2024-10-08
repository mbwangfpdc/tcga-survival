from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
from joblib import Parallel, delayed, parallel_config
from dataprep import *
from typing import Iterable
import pandas as pd
import numpy as np
import time
import logging
import sys
import math

def feature_label(feature_subset: Iterable[str]):
    return "_".join(sorted(list(feature_subset)))

def configure_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():  # To prevent adding handlers multiple times
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def join_features(feature_data_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    configure_logger()
    label = feature_label(feature_data_map.keys())
    logging.info(f"START JOIN {label}")
    assert set(feature_data_map.keys()).issubset(ELIGIBLE_FEATURE_TYPES)
    feature_data = get_joined_feature_data(feature_data_map)
    feature_data.columns = feature_data.columns.astype(str)
    logging.info(f"Join {label} finished!")
    return feature_data


# Given a dataframe and a set of indices, split the dataframe into 2 dataframes
# The first has all rows with a index within the set, and the second has all rows with an index not in the set
def split_df(df: pd.DataFrame, indices: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df.index.isin(indices)], df[~df.index.isin(indices)]


def train_cph(
    outcomes: pd.DataFrame,
    feature_data: pd.DataFrame,
    test_fold: set[str],  # All indices not in this set are training samples
) -> tuple[CoxPHSurvivalAnalysis, float]:
    configure_logger()
    logging.info("START TRAINING")
    test_outcomes, train_outcomes = split_df(outcomes, test_fold)
    test_feature_data, train_feature_data = split_df(feature_data, test_fold)

    start = time.time()
    estimator = CoxPHSurvivalAnalysis(alpha=0.01)
    estimator.fit(train_feature_data, train_outcomes.to_records(index=False))
    score = estimator.score(test_feature_data, test_outcomes.to_records(index=False))
    logging.info(f"Finished training after {time.time() - start} seconds, score is {score}")
    return estimator, score

def generate_join_feature_calls(feature_subsets: list[set[str]]):
    for feature_subset in feature_subsets:
        yield delayed(join_features)({ft: feature_data_map[ft] for ft in feature_subset})


if __name__ == "__main__":
    configure_logger()
    set_config(display="text")  # displays text representation of estimators

    ELIGIBLE_FEATURE_TYPES = set(["expr", "text", "hist_mean", "hist_max"])
    N_FOLDS = 5
    FEATURES_TO_KEEP = 256

    clin_data, feature_data_map = load_raw_data(ELIGIBLE_FEATURE_TYPES)
    clin_data, feature_data_map = harmonize_and_clean(
        clin_data=clin_data, feature_data_map=feature_data_map
    )
    for ft in ELIGIBLE_FEATURE_TYPES:
        feature_data_map[ft] = pca_feature(feature_data_map[ft], FEATURES_TO_KEEP)

    folds = [
        set(subset)
        for subset in np.array_split(
            clin_data.sample(frac=1, random_state=0).index.to_numpy(), N_FOLDS
        )
    ]

    outcomes = clin_data[["days_to_death", "days_to_last_follow_up"]].copy()
    outcomes["days_to_event"] = outcomes[["days_to_death", "days_to_last_follow_up"]].max(
        axis=1
    )
    outcomes["death_witnessed"] = outcomes["days_to_death"] != -1
    # retain only interesting cols
    outcomes = outcomes[["death_witnessed", "days_to_event"]]
    logging.info(
        f"{len(outcomes[outcomes['death_witnessed']])} deaths witnessed out of {len(outcomes)} total samples"
    )

    # Feature sets to try
    feature_subsets = [
        set(subset)
        for subset in [
            ["expr"],
            # ["text"],
            # ["hist_mean"],
            # ["hist_max"],
            ["expr", "text"],
            # ["expr", "hist_mean"],
            # ["expr", "hist_max"],
            # ["text", "hist_mean"],
            # ["text", "hist_max"],
            # ["expr", "text", "hist_mean"],
            # ["expr", "text", "hist_max"],
        ]
    ]

    logging.info("initializing loky backend...")
    with parallel_config(backend="loky", n_jobs=-1):
        logging.info(f"Loky ready. Merging features for each requested feature subset (feature subsets requested: {feature_subsets})")
        # For each feature subset requested, join them.
        joined_data = Parallel()(delayed(join_features)({ft: feature_data_map[ft] for ft in feature_subset}) for feature_subset in feature_subsets)
        logging.info("Feature merging done.")

        train_inputs = []
        for feature_subset, feature_data in zip(feature_subsets, joined_data):
            for i, fold in enumerate(folds):
                train_inputs.append((feature_subset, feature_data, i, fold))
        logging.info(f"Training with {len(feature_subsets)} feature subsets and {N_FOLDS} folds per subset for a total of {len(train_inputs)} jobs on {os.cpu_count()} logical cpus")
        output_gen = Parallel(return_as="generator")(delayed(train_cph)(outcomes, feature_data, fold) for (_, feature_data, _, fold) in train_inputs)

        for (feature_subset, _, i, _), (model, score) in zip(train_inputs, output_gen):
            logging.info(f"{feature_subset}_{i}_{score}")
