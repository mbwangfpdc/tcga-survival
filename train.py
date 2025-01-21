import pandas as pd
import numpy as np
import logging
import time
import os
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
from sksurv.linear_model import CoxPHSurvivalAnalysis, IPCRidge
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis, ExtraSurvivalTrees
from sksurv.functions import StepFunction
from sksurv.metrics import as_concordance_index_ipcw_scorer, concordance_index_ipcw
from sksurv.meta import EnsembleSelection
from typing import Generator
from argparse import Namespace
from joblib import Parallel, delayed
from globals import OUTPUT_PATH
from dataprep import join_features, pca_feature_data

from utils import configure_logger, feature_set_label
from results import ModelResults

SINGLE_THREADED_MODELS = set(["cox", "gb", "cgb", "ipcr"])

class TrainInputs:
    def __init__(
        self,
        outcomes: pd.DataFrame,
        data: pd.DataFrame,
        model_id: str,
    ):
        self.outcomes = outcomes
        self.data = data
        self.model_id = model_id


# Dispatch the training data to appropriate models
# This function returns a generator for the c-index of each model
def dispatch_train(inputs: list[TrainInputs], args: Namespace) -> Generator[SurvivalAnalysisMixin | BaseEstimator, None, None]:
    if args.model in SINGLE_THREADED_MODELS:
        runner = train_parallel
    else:
        runner = train_sequential
    return runner(inputs, args)


def make_estimator(args: Namespace) -> SurvivalAnalysisMixin | BaseEstimator:
    if args.model == "cox":
        return CoxPHSurvivalAnalysis(alpha=args.alpha, n_iter=args.n_iter)
    if args.model == "ipcr":
        return IPCRidge(random_state=args.rand_state)
    elif args.model == "rf":
        return RandomSurvivalForest(
            n_estimators=args.n_est, n_jobs=-3, random_state=args.rand_state
        )
    elif args.model == "est":
        return ExtraSurvivalTrees(
            n_estimators=args.n_est, n_jobs=-3, random_state=args.rand_state
        )
    elif args.model == "gb":
        return GradientBoostingSurvivalAnalysis(
            n_estimators=args.n_est, random_state=args.rand_state
        )
    elif args.model == "cgb":
        return ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=args.n_est, random_state=args.rand_state
        )
    else:
        # Unreachable
        assert False


def train_parallel(inputs: list[TrainInputs], args: Namespace):
    return Parallel(return_as="generator")(
        delayed(train_model)(
            inp,
            args,
        )
        for inp in inputs
    )


def train_sequential(inputs: list[TrainInputs], args: Namespace):
    for inp in inputs:
        yield train_model(inp, args)


def train_model(
    inp: TrainInputs,
    args: Namespace,
) -> SurvivalAnalysisMixin | BaseEstimator:
    configure_logger()
    logging.debug(f"[{inp.model_id}]: Start")
    # The below doesn't work because the anndata format drops column names
    # TODO: normalize age better
    # if "age_at_diagnosis" in train_data:
    #     logging.info("z-score normalizing age")
    #     # We use the mean and std of the training data for both train and test data to avoid data leakage
    #     mean = train_data["age_at_diagnosis"].mean()
    #     std = train_data["age_at_diagnosis"].std()
    #     test_data["age_at_diagnosis"] = (
    #         test_data["age_at_diagnosis"] - mean
    #     ) / std
    #     train_data["age_at_diagnosis"] = (
    #         train_data["age_at_diagnosis"] - mean
    #     ) / std
    start = time.time()
    # TODO: cat ensembling needs to be moved out of the training region to expose test data to the PCA
    # if args.ensemble == "cat" and len(inp.train_data) > 0:
    #     train_data = join_features(inp.train_data)
    #     if args.pca_post_join > 0:
    #         train_data, test_data = pca_feature_data(
    #             train_data, test_data, args.pca_post_join, args
    #         )
    #     inp.train_data = {feature_id(inp.train_data.keys()): train_data}
    # predictions = []
    estimator = as_concordance_index_ipcw_scorer(make_estimator(args))
    estimator.fit(inp.data, inp.outcomes)
    logging.debug(f"[{inp.model_id}] Finished after {time.time() - start} seconds")
    return estimator


# def save_models(blah):
#     if args.save_models:
#         feature_data = pd.concat(inp.train_data, inp.test_data)
#         risk = estimator.predict(feature_data)
#         logging.debug(f"[{inp.model_id}] Predicted risk")

#         def stepfunc_to_array(stepfunc: StepFunction) -> np.ndarray[float]:
#             min_x, max_x = stepfunc.domain
#             return np.fromiter(
#                 (stepfunc(x) for x in np.linspace(min_x, max_x, args.time_steps)),
#                 dtype=float,
#                 count=args.time_steps,
#             )

#         cum_hazard = np.array(
#             list(
#                 map(
#                     stepfunc_to_array,
#                     estimator.predict_cumulative_hazard_function(feature_data),
#                 )
#             )
#         )
#         logging.debug(f"[{inp.model_id}] Predicted cumulative hazards")
#         survival = np.array(
#             list(
#                 map(
#                     stepfunc_to_array, estimator.predict_survival_function(feature_data)
#                 )
#             )
#         )
#         logging.debug(f"[{inp.model_id}] Predicted survival functions")

#         model_results = ModelResults(
#             score=score,
#             duration=train_duration,
#             weights=estimator.coef_,
#             index=feature_data.index,
#             risk=risk,
#             cum_haz=cum_hazard,
#             surv=survival,
#         )
#         model_results.dump(os.path.join(OUTPUT_PATH, args.rundir, f"{inp.model_id}.json"))
#     return score
