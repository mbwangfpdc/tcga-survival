from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
from joblib import Parallel, delayed
from dataprep import *
import time
import logging
import sys

set_config(display="text")  # displays text representation of estimators

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout, level=logging.DEBUG
)

ELIGIBLE_FEATURE_TYPES = set(["expr", "text", "hist_mean", "hist_max"])
N_FOLDS = 5
FEATURES_TO_KEEP = 256

clin_data, feature_data_map = load_raw_data(ELIGIBLE_FEATURE_TYPES)
clin_data, feature_data_map = preharmonize_and_clean(clin_data=clin_data, feature_data_map=feature_data_map)
for ft in ELIGIBLE_FEATURE_TYPES:
    feature_data_map[ft] = pca_feature(feature_data_map[ft], FEATURES_TO_KEEP)

outcomes = clin_data[["days_to_death", "days_to_last_follow_up"]].copy()
outcomes["days_to_event"] = outcomes[["days_to_death", "days_to_last_follow_up"]].max(axis=1)
outcomes["death_witnessed"] = outcomes["days_to_death"] != -1
logging.info(f"{len(outcomes[outcomes['death_witnessed']])} deaths witnessed out of {len(outcomes)} total samples")

# TODO: make index partitions here

# Feature sets to try
feature_subsets = [set(subset) for subset in [
    ["expr"],
    ["text"],
    ["hist_mean"],
    ["hist_max"],
    ["expr", "text"],
    ["expr", "hist_mean"],
    ["expr", "hist_max"],
    ["text", "hist_mean"],
    ["text", "hist_max"],
    ["expr", "text", "hist_mean"],
    ["expr", "text", "hist_max"],
]]

def train_cph(feature_subset: set[str], outcomes: pd.DataFrame, feature_data_map: dict[str, pd.DataFrame]) -> tuple[CoxPHSurvivalAnalysis, float]:
    assert feature_subset.issubset(ELIGIBLE_FEATURE_TYPES)
    feature_data = get_joined_feature_data({ft: feature_data_map[ft] for ft in feature_subset})
    feature_data.columns = feature_data.columns.astype(str)
    outcomes_record = outcomes[["death_witnessed", "days_to_event"]].to_records(index=False)

    # TODO: K-fold cross validation with the different sets

    logging.info("Training with features %s", feature_subset)
    start = time.time()
    estimator = CoxPHSurvivalAnalysis(alpha=0.01)
    estimator.fit(feature_data, outcomes_record)
    logging.info(f"Finished training after {time.time() - start} seconds")
    score = estimator.score(feature_data, outcomes_record)
    logging.info(f"Score: {score}")
    return estimator, score

output_gen = Parallel(n_jobs=-1, return_as="generator")(delayed(train_cph)(feature_subset, outcomes, feature_data_map) for feature_subset in feature_subsets)

for model, score in output_gen:
    logging.info(score)
