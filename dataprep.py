from sklearn.decomposition import PCA
from globals import *
import pandas as pd
import numpy as np
import anndata
import logging

# Returns clinical data as well as a map of feature type to feature data.
# The requested feature data is given by the argument, feature_types.
# If feature_types is not provided, all feature data is used.
def load_raw_data(feature_types: list[str] = []) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # We can arbitrarily drop duplicates since the rows duplicated on case id are the same for our purposes
    logging.info("Reading clinical data from disk...")
    clin_data = pd.read_csv(CLINICAL_TSV_PATH, sep="\t").drop_duplicates(subset=['case_submitter_id']).set_index('case_submitter_id')
    raw_feature_data_map = {}
    for feature_type in feature_types if feature_types else ELIGIBLE_FEATURE_TYPES:
        logging.info("Reading feature %s from disk...", feature_type)
        feature_anndata = anndata.read_h5ad(feature_path_for(feature_type))
        # Hack to extract the pooling type from hist. The variable name for this
        # feature vector is just hist, even though the files are named by the pooling type
        # This hack makes it so feature names cannot contain an _, unless they mean
        # something like pooling. Sorry everyone lol
        raw_feature_data_map[feature_type] = pd.DataFrame(feature_anndata.obsm[f"X_{feature_type.split('_')[0]}"], index=feature_anndata.obs.index)
    return clin_data, raw_feature_data_map

# Apply PCA to extract the N most significant features of a dataframe
def pca_feature(feature_data: pd.DataFrame, features_to_keep: int, normalize: bool = True) -> pd.DataFrame:
    if features_to_keep <= 0 or features_to_keep >= feature_data.shape[1]:
        return
    logging.info(f"Using PCA to reduce {feature_data.shape[1]} dimensions to {features_to_keep}")
    feature_data = pd.DataFrame(PCA(features_to_keep).fit_transform(feature_data), index=feature_data.index)
    if normalize:
        logging.info("Normalizing the feature to the unit vector")
        feature_data = feature_data.apply(lambda x: x / np.linalg.norm(x), axis=1)
    return feature_data

# Join all feature data into a single dataframe
def get_joined_feature_data(feature_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    joined_feature_data = None
    logging.info("Joining features %s...", feature_map.keys())
    for i, (feature_type, feature_data) in enumerate(feature_map.items()):
        if joined_feature_data is None:
            joined_feature_data = feature_data
            logging.info("Initial feature data of type %s is shape %s", feature_type, feature_data.shape)
        else:
            joined_feature_data = joined_feature_data.join(feature_data, how="inner", lsuffix=f"_{i}")
            logging.info("Joined prepared feature data of type %s to make shape %s", feature_type, joined_feature_data.shape)
    return joined_feature_data

# Return a harmonized and cleaned version of clin_data and feature_data
# After this:
#   * Invalid clinical rows where outcomes are not usable will be removed
#   * Rows which are not in ALL feature/clin dataframes will be removed. This means we can treat all feature subsets as coming from the same data.
#   * Feature data columns are strings
def harmonize_and_clean(clin_data: pd.DataFrame, feature_data_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    clin_data[["days_to_death", "days_to_last_follow_up"]] = clin_data[["days_to_death", "days_to_last_follow_up"]].replace("'--", "-1").astype(float).astype(int)
    invalid_outcomes = clin_data[(clin_data["days_to_death"] < 0) & (clin_data["days_to_last_follow_up"] < 0)].index
    for feature_type in feature_data_map:
        feature_data_map[feature_type] = feature_data_map[feature_type].drop(invalid_outcomes, errors="ignore")
    clin_data = clin_data.drop(invalid_outcomes, errors="ignore")
    logging.info(f"Filtered out {len(invalid_outcomes)} invalid clinical cases, {len(clin_data)} remaining")

    # We only care about rows we have both feature and clinical data for
    index_intersection = clin_data.index
    logging.info("Clin data has %s rows", len(clin_data.index))
    for feature_type, feature_data in feature_data_map.items():
        logging.info("%s data has %s rows", feature_type, len(feature_data.index))
        index_intersection = index_intersection.intersection(feature_data.index)
        logging.info("cumulative intersection has %s rows", len(index_intersection))
    clin_data = clin_data[clin_data.index.isin(index_intersection)].sort_index()
    for feature_type, feature_data in feature_data_map.items():
        feature_data_map[feature_type] = feature_data[feature_data.index.isin(index_intersection)].sort_index()
        assert len(clin_data) == len(feature_data_map[feature_type])
    logging.info(f"The intersection of clinical and feature data is of size {len(clin_data)}")

    return clin_data, feature_data_map
