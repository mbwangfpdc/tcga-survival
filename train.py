import pandas as pd
import numpy as np
import logging
import time
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
from sksurv.linear_model import CoxPHSurvivalAnalysis, IPCRidge
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis, ExtraSurvivalTrees
from sksurv.functions import StepFunction
from typing import Generator, List
from argparse import Namespace
from joblib import Parallel, delayed

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
    start = time.time()
    estimator = make_estimator(args)
    estimator.fit(inp.data, inp.outcomes)
    logging.debug(f"[{inp.model_id}] Finished after {time.time() - start} seconds")
    return estimator

# Use tolist to convert any
def tolist(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not hasattr(result, "tolist"):
            assert False # Bad, bad person! Only annotate functions that return numpy types with tolist!
        return result.tolist()
    return wrapper

def predict_all(sfs: List[StepFunction], unique_times: List[float]) -> List[List[float]]:
    return [sf(unique_times).tolist() for sf in sfs]

# Take some already-trained models and implement an ensemble of them
class EnsembleModel():
    def __init__(self, models: List[BaseEstimator], args: Namespace):
        self.models = models
        self.metamodel = None
        self.args = args

    # Return an ndarray where each rows are cases and cols are a model's prediction for that case
    def _get_predictions(self, data: List[pd.DataFrame]) -> np.ndarray:
        assert len(self.models) == len(data)
        return np.array(
            [model.predict(data) for model, data in zip(self.models, data)]
        ).transpose()

    # Return an ndarray where each rows are cases and cols are a model's predicted survival curve for that case
    def _get_predicted_survival_curve(self, data: List[pd.DataFrame], unique_times: List[float]) -> np.ndarray:
        assert len(self.models) == len(data)
        return np.array(
            [predict_all(model.predict_survival_function(data), unique_times) for model, data in zip(self.models, data)]
        ).transpose(1, 0, 2)

    def train(self, train_data: List[pd.DataFrame], train_outcomes: np.rec.recarray):
        assert len(self.models) == len(train_data)
        if len(self.models) == 1:
            return
        if self.args.ensemble in ["cox", "ipcr"]:
            train_predictions = self._get_predictions(train_data)
            if self.args.ensemble == "cox":
                self.metamodel = CoxPHSurvivalAnalysis()
                self.metamodel.fit(train_predictions, train_outcomes)
            elif self.args.ensemble == "ipcr":
                self.metamodel = IPCRidge(random_state=self.args.rand_state)
                self.metamodel.fit(train_predictions, train_outcomes)

    def predict(self, data: List[pd.DataFrame]) -> List[float]:
        assert len(self.models) == len(data)
        if len(self.models) == 1:
            return self.models[0].predict(data[0])
        else:
            predictions = self._get_predictions(data)
            if self.args.ensemble == "mean":
                return np.mean(predictions, axis=1)
            elif self.args.ensemble == "meanrank":
                ranks = np.apply_along_axis(rankdata, 0, predictions)
                return np.mean(ranks, axis=1)
            elif self.args.ensemble in ["cox", "ipcr"]:
                return self.metamodel.predict(predictions)
            else:
                assert False

    def predict_survival_function(self, data: List[pd.DataFrame], unique_times: List[float]) -> List[List[float]]:
        assert len(self.models) == len(data)
        if len(self.models) == 1:
            return predict_all(self.models[0].predict_survival_function(data[0]), unique_times)
        else:
            if self.args.ensemble == "mean":
                predictions = self._get_predicted_survival_curve(data, unique_times)
                return np.mean(predictions, axis=1)
            # elif self.args.ensemble == "meanrank": # meanrank cannot be done along 2d data
                # ranks = np.apply_along_axis(rankdata, 0, predictions)
                # predictions = np.mean(ranks, axis=1)
            elif self.args.ensemble == "cox" or self.args.ensemble == "ipcr":
                predictions = self._get_predictions(data)
                return predict_all(self.metamodel.predict_survival_function(predictions), unique_times)
            else:
                assert False

