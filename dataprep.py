from sklearn.decomposition import PCA
from globals import ELIGIBLE_FEATURE_TYPES, CLINICAL_TSV_PATH, DATA_PATH
from utils import configure_logger, feature_id
from argparse import Namespace
import pandas as pd
import numpy as np
import anndata
import logging
import os


def feature_path_for(feature_type: str) -> str:
    return os.path.join(DATA_PATH, f"X_{feature_type}.h5ad")


# Returns clinical data as well as a map of feature type to feature data.
# The requested feature data is given by the argument, feature_types.
# If feature_types is not provided, all feature data is used.
def load_raw_data(
    feature_types: set[str] = set(),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # We can arbitrarily drop duplicates since the rows duplicated on case id are the same for our purposes
    logging.debug("Reading clinical data from disk...")
    clin_data = (
        pd.read_csv(CLINICAL_TSV_PATH, sep="\t")
        .drop_duplicates(subset=["case_submitter_id"])
        .set_index("case_submitter_id")
    )
    raw_feature_data_map = {}
    for feature_type in feature_types if feature_types else ELIGIBLE_FEATURE_TYPES:
        logging.debug("Reading feature %s from disk...", feature_type)
        feature_anndata = anndata.read_h5ad(feature_path_for(feature_type))
        feature_df = pd.DataFrame(
            feature_anndata.obsm[f"X_{feature_type}"],
            index=feature_anndata.obs.index,
        )
        raw_feature_data_map[feature_type] = feature_df
        logging.debug("Feature %s has %s rows and %s columns", feature_type, len(feature_df), len(feature_df.columns))
    return clin_data, raw_feature_data_map


# Join all feature data into a single dataframe
def get_joined_feature_data(feature_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    joined_feature_data = None
    logging.debug("Joining features %s...", feature_map.keys())
    for i, (feature_type, feature_data) in enumerate(feature_map.items()):
        if joined_feature_data is None:
            joined_feature_data = feature_data
            logging.debug(
                "Initial feature data of type %s is shape %s",
                feature_type,
                feature_data.shape,
            )
        else:
            joined_feature_data = joined_feature_data.join(
                feature_data, how="inner", lsuffix=f"_{i}"
            )
            logging.debug(
                "Joined prepared feature data of type %s to make shape %s",
                feature_type,
                joined_feature_data.shape,
            )
    return joined_feature_data


# Calculate and extract columns relevant for sksurv
def prepare_outcomes(clin_data: pd.DataFrame) -> pd.DataFrame:
    outcomes = pd.DataFrame()
    # clin_data[["days_to_death", "days_to_last_follow_up", "vital_status"]].copy()
    outcomes["days_to_event"] = clin_data[
        ["days_to_death", "days_to_last_follow_up"]
    ].max(axis=1)
    outcomes["death_witnessed"] = clin_data["vital_status"] == "Dead"
    outcomes["death_witnessed"] = clin_data["days_to_death"] != -1
    logging.debug(
        f"{len(outcomes[outcomes['death_witnessed']])} deaths witnessed out of {len(outcomes)} total samples"
    )
    return outcomes[["death_witnessed", "days_to_event"]]


# Return a harmonized and cleaned version of clin_data and feature_data
# After this:
#   * Invalid clinical rows where outcomes are not usable will be removed
#   * Rows which are not in ALL feature/clin dataframes will be removed. This means we can treat all feature subsets as coming from the same data.
#   * Feature data columns are strings
def harmonize_and_clean(
    clin_data: pd.DataFrame, feature_data_map: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    import re
    regex = re.compile(r"^('--|unknown|not reported)$", re.IGNORECASE)
    clin_data.replace(regex, "'--", inplace=True, regex=True)
    clin_data[["days_to_death", "days_to_last_follow_up"]] = (
        clin_data[["days_to_death", "days_to_last_follow_up"]]
        .replace("'--", "-1")
        .astype(float)
        .astype(int)
    )
    ok_outcome_predicate = (
        (clin_data["vital_status"] == "Alive") &
        (clin_data["days_to_death"] < 0) &
        (clin_data["days_to_last_follow_up"] >= 0)
    ) | (
        (clin_data["vital_status"] == "Dead") &
        (clin_data["days_to_death"] >= 0) &
        (clin_data["days_to_last_follow_up"] <= clin_data["days_to_death"])
    )
    invalid_outcomes_index = clin_data[~ok_outcome_predicate].index
    # TODO: Tell steven about this, probably revert? Save the rows
    # invalid_outcomes_index = clin_data[
    #     (clin_data["days_to_death"] < 0) & (clin_data["days_to_last_follow_up"] < 0)
    # ].index
    # print(clin_data[["days_to_death", "days_to_last_follow_up", "vital_status"]].loc[invalid_outcomes_index.difference(invalid_outcomes_index_2)])
    # # print(clin_data[["days_to_death", "days_to_last_follow_up"]].loc[invalid_outcomes_index & ~])
    # exit(1)

    for feature_type in feature_data_map:
        feature_data_map[feature_type] = feature_data_map[feature_type].drop(
            invalid_outcomes_index, errors="ignore"
        )
    clin_data = clin_data.drop(invalid_outcomes_index, errors="ignore")
    logging.debug(
        f"Filtered out {len(invalid_outcomes_index)} invalid clinical cases, {len(clin_data)} remaining"
    )

    # We only care about rows we have both feature and clinical data for
    index_intersection = clin_data.index
    logging.debug("Clin data has %s rows", len(clin_data.index))
    for feature_type, feature_data in feature_data_map.items():
        logging.debug("%s data has %s rows", feature_type, len(feature_data.index))
        index_intersection = index_intersection.intersection(feature_data.index)
        logging.debug("cumulative intersection has %s rows", len(index_intersection))
    clin_data = clin_data[clin_data.index.isin(index_intersection)].sort_index()
    for feature_type, feature_data in feature_data_map.items():
        feature_data_map[feature_type] = feature_data[
            feature_data.index.isin(index_intersection)
        ].sort_index()
        assert len(clin_data) == len(feature_data_map[feature_type])
    logging.debug(
        f"The intersection of clinical and feature data is of size {len(clin_data)}"
    )

    return clin_data, feature_data_map


# Given some feature data, fit a PCA over the training data then transform the training and test data and return them
def pca_feature_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    components: int,
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_logger()
    # train_data = train_data.sort_index()
    # test_data = test_data.sort_index()
    if components > 0 and components < train_data.shape[1]:
        logging.debug(
            f"Using PCA to reduce {train_data.shape[1]} dimensions to {components}"
        )
        pca = PCA(components, random_state=args.rand_state)
        pca.fit(train_data)
        train_data = pd.DataFrame(pca.transform(train_data), index=train_data.index)
        test_data = pd.DataFrame(pca.transform(test_data), index=test_data.index)
    else:
        logging.debug(
            f"Skipping PCA as {components=} is not applicable with {train_data.shape[1]} dimensions"
        )
    logging.debug("Normalizing the transformed feature vectors to the unit vector")
    test_data = test_data.apply(lambda x: x / np.linalg.norm(x), axis=1)
    train_data = train_data.apply(
        lambda x: x / np.linalg.norm(x), axis=1
    )
    return train_data, test_data

def join_train_test_features(train_test_feature_data_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]], args: Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = {}
    test_features = {}
    for ft, (train, test) in train_test_feature_data_map.items():
        train_features[ft] = train
        test_features[ft] = test
    train_data, test_data = join_features(train_features), join_features(test_features)
    if args.pca_post_join > 0:
        train_data, test_data = pca_feature_data(train_data, test_data, args.pca_post_join, args)
    return train_data, test_data


# Columnwise concat feature dataframes
def join_features(feature_data_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    configure_logger()
    fid = feature_id(feature_data_map.keys())
    logging.debug(f"START JOIN {fid}")
    assert set(feature_data_map.keys()).issubset(ELIGIBLE_FEATURE_TYPES)
    feature_data = get_joined_feature_data(feature_data_map)
    feature_data.columns = feature_data.columns.astype(str)
    logging.debug(f"Join {fid} finished!")
    return feature_data

