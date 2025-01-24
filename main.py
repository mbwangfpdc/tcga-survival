#!/usr/bin/env python3

from sklearn import set_config
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, IPCRidge
from sksurv.metrics import (
    concordance_index_ipcw,
    concordance_index_censored,
)
from sksurv.nonparametric import kaplan_meier_estimator
from joblib import Parallel, delayed, parallel_config
from collections import defaultdict
from scipy.stats import rankdata
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import statistics as stats
import time
import logging
import os
import json
import shutil

from dataprep import *
from globals import *
from utils import *
from persist import *
from train import dispatch_train, TrainInputs, SINGLE_THREADED_MODELS
from parser import parse


def calculate_c_index(
    args: Namespace, train_outcomes: pd.DataFrame, test_outcomes: pd.DataFrame, predictions: np.ndarray
) -> float:
    if args.cindex == "harrel":
        return concordance_index_censored(
            test_outcomes["death_witnessed"],
            test_outcomes["days_to_event"],
            predictions,
        )[0]
    elif args.cindex == "uno":
        return concordance_index_ipcw(
            train_outcomes,
            test_outcomes,
            predictions,
        )[0]
    else:
        assert False # unreachable

def main():
    args = parse()
    start = time.time()
    configure_logger()
    set_config(display="text")  # displays text representation of estimators

    run_dir = os.path.join(OUTPUT_PATH, args.rundir)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    logging.info(f"Created run directory at {run_dir}")

    # All data is always loaded from disk to make the datasets between modalities identical
    clin_data, raw_feature_data_map = load_raw_data(ELIGIBLE_FEATURE_TYPES)
    clin_data, raw_feature_data_map = harmonize_and_clean(
        clin_data=clin_data, feature_data_map=raw_feature_data_map
    )
    for feature_type in list(raw_feature_data_map.keys()):
        if feature_type not in args.features and feature_type not in args.solo_features:
            del raw_feature_data_map[feature_type]

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.rand_state)
    folds = [
        train_test
        for train_test in skf.split(
            clin_data, clin_data["project_id"].astype("category").cat.codes
        )
    ]
    logging.info(f"Splitting data into {args.folds} folds...")
    outcomes_df = prepare_outcomes(clin_data)
    train_outcomes = []
    test_outcomes = []
    for train, test in folds:
        train_outcomes.append(outcomes_df.iloc[train].to_records(index=False))
        test_outcomes.append(outcomes_df.iloc[test].to_records(index=False))
    # folded_outcomes = [
    #     (outcomes.iloc[train], outcomes.iloc[test]) for train, test in folds
    # ]
    train_feature_data = []
    test_feature_data = []
    # folded_feature_data = []
    for train, test in folds:
        train_map = {}
        test_map = {}
        for feature_type, feature_data in raw_feature_data_map.items():
            train_map[feature_type] = feature_data.iloc[train]
            test_map[feature_type] = feature_data.iloc[test]
        train_feature_data.append(train_map)
        test_feature_data.append(test_map)
    # Feature sets to try.
    feature_subsets: list[set[str]] = get_feature_sets(args.features) + [
        set([f]) for f in args.solo_features
    ]
    feature_subsets.sort(key=feature_set_label)

    # TODO: Fold arrangement sucks, find a better way to lay it out. Maybe just a big forloop so the code doesn't have to think about the folds, and fold/unfold before and after. Less performant (more barriers) but nicer to look at.
    logging.debug("initializing loky backend...")
    with parallel_config(backend="loky", n_jobs=-3):
        if args.pca_pre_join > 0:
            logging.info(
                f"Using PCA to reduce features to {args.pca_pre_join} most important dimensions..."
            )
            pca_inputs = [
                (fold, ft) for fold, fdm in enumerate(test_feature_data) for ft in fdm
            ]
            reducer_output = Parallel(
                return_as="generator", n_jobs=min(os.cpu_count() - 2, len(pca_inputs))
            )(
                delayed(pca_feature_data)(
                    train_feature_data[fold][ft],
                    test_feature_data[fold][ft],
                    args.pca_pre_join,
                    args,
                )
                for fold, ft in pca_inputs
            )
            for (fold, ft), (train, test) in zip(pca_inputs, reducer_output):
                train_feature_data[fold][ft] = train
                test_feature_data[fold][ft] = test

        if args.ensemble == "cat":

            def cat_unimodal_data(data_maps: list[dict[str, pd.DataFrame]]):
                for fold in range(args.folds):
                    for feature_subset in feature_subsets:
                        if len(feature_subset) > 1:
                            fid = feature_set_label(feature_subset)
                            data_maps[fold][fid] = join_features(
                                {ft: data_maps[fold][ft] for ft in feature_subset}
                            )

            cat_unimodal_data(train_feature_data)
        train_inputs: list[TrainInputs] = []
        for fold, feature_data_map in enumerate(train_feature_data):
            for feature, feature_data in feature_data_map.items():
                train_inputs.append(
                    (
                        TrainInputs(
                            outcomes=train_outcomes[fold],
                            data=feature_data,
                            model_id=f"{feature}_{fold}",
                        ),
                        fold,
                        feature,
                    )
                )
        # for fold, fdm in enumerate(folded_feature_data):
        #     for feature_subset in feature_subsets:
        #         train_inputs.append(
        #             TrainInputs(
        #                 train_outcomes=folded_outcomes[fold][0],
        #                 test_outcomes=folded_outcomes[fold][1],
        #                 train_data={k: fdm[k][0] for k in feature_subset},
        #                 test_data={k: fdm[k][1] for k in feature_subset},
        #                 model_id=f"{feature_id(feature_subset)}_{fold}",
        #             )
        #         )
        # We want to schedule longer single-threaded models first,
        # and shorter multi-threaded models first
        long_model_first = args.model in SINGLE_THREADED_MODELS
        train_inputs.sort(
            key=lambda inp_tuple: inp_tuple[0].data.shape[1],
            reverse=long_model_first,
        )
        logging.info(
            f"Training with {len(feature_subsets)} feature sets and {args.folds} folds per subset for a total of {len(train_inputs)} models"
        )
        output_gen = dispatch_train(
            inputs=list(map(lambda inp_tuple: inp_tuple[0], train_inputs)),
            args=args,
        )

        models: List[Dict[str, CoxPHSurvivalAnalysis]] = [{} for _ in range(args.folds)]
        for (_, fold, featureset), model in zip(train_inputs, output_gen):
            models[fold][featureset] = model

        # TODO: Add cli argument to disable this
        model_map: Dict[str, List[CoxPHSurvivalAnalysis]] = defaultdict(lambda: [0] * args.folds)
        for fold, models_ in enumerate(models):
            for model_type, model in models_.items():
                model_map[model_type][fold] = model
        scores: List[Dict[str, float]] = [{} for _ in range(args.folds)]
        if args.ensemble == "cat":
            # TODO: Relieve memory pressure by dropping training data
            train_inputs = []
            train_feature_data = []
            # Now, as late as possible, cat test data for evaluation
            cat_unimodal_data(test_feature_data)
            for fold, models_ in enumerate(models):
                for featureset, model in models_.items():
                    scores[fold][featureset] = model.score(
                        test_feature_data[fold][featureset], test_outcomes[fold]
                    )
        elif args.ensemble in ["mean", "meanrank", "cox", "ipcr"]:
            for fold, models_ in enumerate(models):
                if args.ensemble in ["cox", "ipcr"]:
                    train_pred_map = {
                            feature: model.predict(train_feature_data[fold][feature])
                            for feature, model in models_.items()
                        }
                pred_map = {
                    feature: model.predict(test_feature_data[fold][feature])
                    for feature, model in models_.items()
                }
                for feature_subset in feature_subsets:
                    label = feature_set_label(feature_subset)
                    if len(feature_subset) == 1:
                        predictions = pred_map[next(iter(feature_subset))]
                    else:
                        predictions = np.array(
                            [pred_map[f] for f in feature_subset]
                        ).transpose()
                        if args.ensemble in ["cox", "ipcr"]:
                            train_predictions = np.array(
                                [train_pred_map[f] for f in feature_subset]
                            ).transpose()
                        if args.ensemble == "mean":
                            predictions = np.mean(predictions, axis=1)
                        elif args.ensemble == "meanrank":
                            ranks = np.apply_along_axis(rankdata, 0, predictions)
                            predictions = np.mean(ranks, axis=1)
                        elif args.ensemble == "cox":
                            metamodel = CoxPHSurvivalAnalysis()
                            # TODO: should we parallelize this?
                            metamodel.fit(train_predictions, train_outcomes[fold])
                            model_map[label][fold] = metamodel
                            predictions = metamodel.predict(predictions)
                        elif args.ensemble == "ipcr":
                            metamodel = IPCRidge(random_state=args.rand_state)
                            # TODO: should we parallelize this?
                            metamodel.fit(train_predictions, train_outcomes[fold])
                            model_map[label][fold] = metamodel
                            predictions = metamodel.predict(predictions)
                    scores[fold][label] = calculate_c_index(
                        args,
                        train_outcomes[fold],
                        test_outcomes[fold],
                        predictions,
                    )  # TODO: is there any use for the other components? maybe consider saving
        else:
            # Unreachable
            assert False
        with open(os.path.join(run_dir, "run.json"), "w+") as run_json:
            run = {}
            run["args"] = vars(args)
            run["scores"] = defaultdict(lambda: [0] * args.folds)
            for fold, score_map in enumerate(scores):
                for model_type, score in score_map.items():
                    run["scores"][model_type][fold] = score
            for model_type, scores in run["scores"].items():
                logging.info(f"{model_type},{stats.mean(scores)},{stats.stdev(scores)}")
            json.dump(run, run_json)
        for model_type, models in model_map.items():
            save_models(models, run_dir, model_type)
        logging.info(f"Finished in {time.time() - start} seconds total.")
        logging.info(f"Rundir was {run_dir}")
        # NOTE: All this code is currently controlled by getting commented out or not. Sue me.
        # TODO: Multimodal guys?
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        # Map of model type -> case -> survival curve predicted
        corrected_report_curves: List[Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = [defaultdict(dict) for _ in range(args.folds)]
        for model_type, models_ in model_map.items():
            if "text" not in model_type:
                continue
            assert not "-" in model_type
            # As of now, we're only creating figures from fold 0
            model = models_[0]
            test_df = test_feature_data[0][model_type]
            test_df: pd.DataFrame = test_df.loc[CORRECTED_REPORTS]
            curves = model.predict_survival_function(test_df, return_array=True).tolist()
            for index, curve in zip(test_df.index, curves, strict=True):
                corrected_report_curves[0][model_type][index] = (model.unique_times_, curve)
            # corrected_report_curves[fold][model_type] = model.predict_cumulative_hazard_function(test_df, return_array=True).tolist()
            logging.debug(f"Added corrected reports for {model_type=} of size {len(test_df)}")
            cindex = model.score(test_df, outcomes_df.loc[CORRECTED_REPORTS].to_records(index=False))
            logging.info(f"{model_type=} had an in-sample {cindex=}")
        with open(f"curves.json", "w+") as curve_file:
            json.dump(corrected_report_curves, curve_file, cls=NpEncoder)
        def dump_km(km_x, km_y, ci, filename: str):
            with open(filename, "w+") as km_json:
                json.dump({"x": km_x, "y": km_y, "ci": ci}, km_json, cls=NpEncoder)
        km_x, km_y, ci = kaplan_meier_estimator(test_outcomes[0]["death_witnessed"], test_outcomes[0]["days_to_event"], conf_type="log-log")
        dump_km(km_x, km_y, ci, "km_full.json")
        km_x, km_y, ci = kaplan_meier_estimator(outcomes_df.loc[CORRECTED_REPORTS]["death_witnessed"], outcomes_df.loc[CORRECTED_REPORTS]["days_to_event"], conf_type="log-log")
        dump_km(km_x, km_y, ci, "km_sample.json")
        # dump subpopulation estimation

if __name__ == "__main__":
    main()
