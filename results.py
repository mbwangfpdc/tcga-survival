"""
This module helps visualize the results of a run
"""

import os
import globals
import re
import logging
import pickle
import json
import pandas as pd
import numpy as np
from typing import Self
from collections import defaultdict
from utils import model_type_and_fold


class ModelResults:
    def __init__(
        self,
        score: float,
        duration: float,
        weights: np.ndarray[float],
        index: pd.Index,
        risk: np.ndarray[float],
        cum_haz: np.ndarray[float],
        surv: np.ndarray[float],
    ):
        self.score = score
        self.duration = duration
        self.weights = weights
        self.index = index
        self.risk = risk
        self.cum_haz = cum_haz
        self.surv = surv
        assert len(index) == len(risk)
        assert len(risk) == len(cum_haz)
        assert len(cum_haz) == len(surv)

    def dump(self, path: str):
        data = {}
        data["score"] = self.score
        data["duration"] = self.duration
        data["weights"] = self.weights.tolist()
        data["index"] = self.index.tolist()
        data["risk"] = self.risk.tolist()
        data["cum_haz"] = self.cum_haz.tolist()
        data["surv"] = self.surv.tolist()
        with open(path, "w+") as model_json:
            json.dump(data, model_json, indent=4)

    def load(path: str) -> Self:
        with open(path, "r") as model_json:
            data = json.load(model_json)
            return ModelResults(
                score=data["score"],
                duration=data["duration"],
                weights=np.ndarray(data["weights"]),
                index=pd.Index(data["index"]),
                risk=np.ndarray(data["risk"]),
                cum_haz=np.ndarray(data["cum_haz"]),
                surv=np.ndarray(data["surv"]),
            )


# For each model type, aggregate its folds into a average and stdev ModelResults.
def agg_models_by_type(
    model_map: dict[str, ModelResults]
) -> dict[str, tuple[ModelResults, ModelResults]]:
    aggregated = defaultdict(list)
    for model_id, model in model_map:
        # Drop fold identifier during aggregation
        true_id, _ = model_type_and_fold(model_id)
        aggregated[true_id].append(model)
    for model_id, models in aggregated.items():
        avg_score, std_score = mean_std_floats(map(lambda m: m.score, models))
        avg_duration, std_duration = mean_std_floats(map(lambda m: m.duration, models))
        avg_weights, std_weights = mean_std_ndarrays(map(lambda m: m.weights, models))
        index = models[0].index
        avg_risk, std_risk = mean_std_ndarrays(map(lambda m: m.risk, models))
        avg_cum_haz, std_cum_haz = mean_std_ndarrays(map(lambda m: m.cum_haz, models))
        avg_surv, std_surv = mean_std_ndarrays(map(lambda m: m.surv, models))
        aggregated[model_id] = (
            ModelResults(
                score=avg_score,
                duration=avg_duration,
                weights=avg_weights,
                index=index,
                risk=avg_risk,
                cum_haz=avg_cum_haz,
                surv=avg_surv,
            ),
            ModelResults(
                score=std_score,
                duration=std_duration,
                weights=std_weights,
                index=index,
                risk=std_risk,
                cum_haz=std_cum_haz,
                surv=std_surv,
            ),
        )


# Given a list of ndarrays of equal size, return two ndarrays of that size.
# The first contains the mean of every value, the second contains the stdev.
def mean_std_ndarrays(
    ndarrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    collected = np.array(ndarrays)
    return collected.mean(axis=0), collected.std(axis=0)


def mean_std_floats(floats: list[float]) -> tuple[float, float]:
    ndarray = np.array(floats)
    return ndarray.mean(), ndarray.std()


class ExperimentResults:
    def __init__(self, model_results: list[ModelResults]):
        self.model_results = model_results

    def analyze():
        # TODO
        pass
