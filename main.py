#!/usr/bin/env python3

from sklearn import set_config
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_ipcw,
    concordance_index_censored,
)
from joblib import Parallel, delayed, parallel_config
from typing import List, Dict, Tuple, Callable
import pandas as pd
import numpy as np
import dataclasses as dc
import time
import logging
import os
import json
import shutil
import math

from dataprep import *
from globals import *
from utils import *
from persist import *
from train import dispatch_train, TrainInputs, SINGLE_THREADED_MODELS, EnsembleModel
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

# A run-unique identifier for datasets. Some members distinguish data by row (patient vs. patient) and
# others distinguish data by column (patient attribute vs patient attribute).
@dc.dataclass(eq=True, frozen=True)
class DataKey:
    # Data modality like text, histology, expression, etc. Column-wise.
    feature_type: str = ""
    # Cross-validation fold specifier. Row-wise.
    fold: int = -1
    # Population group specifier. Row-wise.
    # Used when we want to group our models on an additional variable. This will contain
    # a value from a categorical column of the clinical data, such as 'project_id'.
    # This allows testing models on population subsets (e.g. men vs women) as opposed to
    # data subsets (text vs image)
    # Multi-dimensional slicing is not currently supported.
    group: str = ""

# A custom dictionary keyed on DataKey which can map from the given DataKey to
# another DataKey. Should be used to represent many->1 DataKey relationships,
# such as a dict that only reads the fold of the given DataKeys.
class MappingDict(dict):
    def __init__(self, keymap: Callable[[DataKey], DataKey]):
        super().__init__()
        self.keymap = keymap
    def __getitem__(self, key: DataKey):
        assert isinstance(key, DataKey)
        return super().__getitem__(self.keymap(key))
    def __setitem__(self, key: DataKey, value):
        assert isinstance(key, DataKey)
        super().__setitem__(self.keymap(key), value)

def ignoring_feature(key: DataKey) -> DataKey:
    return DataKey(fold=key.fold, group=key.group)
def ignoring_group(key: DataKey) -> DataKey:
    return DataKey(feature_type=key.feature_type, fold=key.fold)
def ignoring_feature_and_group(key: DataKey) -> DataKey:
    return ignoring_feature(ignoring_group(key))

def ensemble_keys(groups: Iterable[str], args: Namespace) -> List[DataKey]:
    return [
        DataKey(feature_type=feature_set_label(feature_set), fold=fold, group=group)
        for fold in range(args.folds)
        for feature_set in args.ensembles
        for group in groups
    ]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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

    def runpath(filename: str) -> str:
        return os.path.join(run_dir, filename)

    # All data is always loaded from disk to make the datasets between modalities identical
    clin_data, raw_feature_data_map = load_raw_data(ELIGIBLE_FEATURE_TYPES)
    clin_data, raw_feature_data_map = harmonize_and_clean(
        clin_data=clin_data, feature_data_map=raw_feature_data_map
    )
    for feature_type in list(raw_feature_data_map.keys()):
        if feature_type not in args.features:
            del raw_feature_data_map[feature_type]

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.rand_state)

    train_groups: Dict[str, pd.Index] = {"": clin_data.index}
    test_groups: Dict[str, pd.Index] = {"": clin_data.index}
    if args.train_group_by:
        train_groups = clin_data.groupby(args.train_group_by).apply(lambda x: x.index, include_groups=False).to_dict()
    if args.test_group_by:
        test_groups = clin_data.groupby(args.test_group_by).apply(lambda x: x.index, include_groups=False).to_dict()
    # # TODO: why is this necessary? understand the PCA restriction
    train_group_and_index = list(train_groups.items())
    for group, index in train_group_and_index:
        # Discard any groups whose data is too small to PCA properly
        if len(index) < (args.pca_pre_join * (args.folds - 1) / args.folds):
            logging.info(f"Discarding {group}")
            del train_groups[group]
            if group in test_groups:
                del test_groups[group]
    # TODO: consider stratifying by whatever the grouped field is instead of always project
    folds: List[Tuple[pd.Index, pd.Index]] = [
        (clin_data.iloc[train].index, clin_data.iloc[test].index)
        for train, test in skf.split(
            clin_data, clin_data["project_id"].astype("category").cat.codes
        )
    ]
    logging.info(f"Splitting data into {args.folds} folds...")
    outcomes_df: pd.DataFrame = prepare_outcomes(clin_data, args)
    train_outcomes: Dict[DataKey, np.rec.recarray] = MappingDict(ignoring_feature) if args.train_group_by else MappingDict(ignoring_feature_and_group)
    test_outcomes: Dict[DataKey, np.rec.recarray] = MappingDict(ignoring_feature) if args.test_group_by else MappingDict(ignoring_feature_and_group)
    logging.info(f"Grouping outcome data...")
    outcomes_df["fold"] = -1
    for fold, (_, test) in enumerate(folds):
        outcomes_df.loc[test, "fold"] = fold
    assert -1 not in outcomes_df["fold"].unique()
    assert np.all((outcomes_df["days_to_event"] >= 0))
    model_extrapolation_limit = math.inf
    for fold, (train, test) in enumerate(folds):
        for group, index in train_groups.items():
            datakey = DataKey(fold=fold, group=group)
            train_outcomes[datakey] = outcomes_to_array(outcomes_df.loc[index.intersection(train)])
            model_extrapolation_limit = min(model_extrapolation_limit, train_outcomes[datakey]["days_to_event"].max())
        for group, index in test_groups.items():
            datakey = DataKey(fold=fold, group=group)
            test_outcomes[datakey] = outcomes_to_array(outcomes_df.loc[index.intersection(test)])

    # If training or test data is not grouped by, ignore group portion of key
    train_feature_data: Dict[DataKey, pd.DataFrame] = {} if args.train_group_by else MappingDict(ignoring_group)
    test_feature_data: Dict[DataKey, pd.DataFrame] = {} if args.test_group_by else MappingDict(ignoring_group)
    logging.info(f"Grouping training and test data...")
    for fold, (train, test) in enumerate(folds):
        for group, index in train_groups.items():
            folded_index = index.intersection(train)
            for feature_type, feature_data in raw_feature_data_map.items():
                datakey = DataKey(feature_type, fold, group)
                train_feature_data[datakey] = feature_data.loc[folded_index]
        for group, index in test_groups.items():
            folded_index = index.intersection(test)
            for feature_type, feature_data in raw_feature_data_map.items():
                datakey = DataKey(feature_type, fold, group)
                test_feature_data[datakey] = feature_data.loc[folded_index]

    with parallel_config(backend="loky", n_jobs=-3):
        pca_models = {} if args.train_group_by else MappingDict(ignoring_group)
        if args.pca_pre_join > 0:
            logging.info(f"Training PCA with {args.pca_pre_join} dimensions...")
            reducer_output = Parallel(
                return_as="generator", n_jobs=min(os.cpu_count() - 2, len(train_feature_data))
            )(
                delayed(train_pca)(
                    data,
                    args.pca_pre_join,
                    args,
                )
                for data in train_feature_data.values()
            )
            for datakey, (train, pca) in zip(train_feature_data.keys(), reducer_output, strict=True):
                train_feature_data[datakey] = train
                pca_models[datakey] = pca
            logging.info(f"Applying PCA to test data...")
            reducer_output = Parallel(
                return_as="generator", n_jobs=min(os.cpu_count() - 2, len(test_feature_data))
            )(
                delayed(apply_pca)(
                    data,
                    pca_models[datakey]
                )
                for datakey, data in test_feature_data.items()
            )
            for datakey, data in zip(test_feature_data.keys(), reducer_output, strict=True):
                test_feature_data[datakey] = data

        if args.ensemble == "cat":

            def cat_unimodal_data(data_maps: Dict[DataKey, pd.DataFrame]):
                for fold in range(args.folds):
                    for feature_subset in args.ensembles:
                        if len(feature_subset) > 1:
                            new_datakey = DataKey(feature_set_label(feature_subset), fold)
                            data_maps[new_datakey] = join_features(
                                {ft: data_maps[DataKey(ft, fold)] for ft in feature_subset}
                            )

            cat_unimodal_data(train_feature_data)

        # This is different from datakeys because that might reference test-only sets. For example,
        # we may be training on a representative set and testing on a specific subset.
        datakeys_for_train: List[DataKey] = list(train_feature_data.keys())
        # We want to schedule longer single-threaded models first,
        # and shorter multi-threaded models first
        long_model_first = args.model in SINGLE_THREADED_MODELS
        datakeys_for_train.sort(
            key=lambda datakey: train_feature_data[datakey].shape[1],
            reverse=long_model_first,
        )
        logging.info(
            f"Training with {len(args.ensembles)} feature sets and {args.folds} folds per subset for a total of {len(datakeys_for_train)} models"
        )
        def train_inputs_from_datakey(datakey: DataKey) -> TrainInputs:
            return TrainInputs(
                outcomes=train_outcomes[datakey],
                data=train_feature_data[datakey],
                model_id=f"{datakey.feature_type}_{datakey.fold}_{datakey.group.replace(FEATURE_ID_DELIM, '_')}",
            )
        output_gen = dispatch_train(
            inputs=list(map(train_inputs_from_datakey, datakeys_for_train)),
            args=args,
        )

        base_models: Dict[DataKey, CoxPHSurvivalAnalysis] = {} if args.train_group_by else MappingDict(ignoring_group)
        for datakey, model in zip(datakeys_for_train, output_gen, strict=True):
            base_models[datakey] = model

        # === SCORING ===

        def new_datakey_df() -> pd.DataFrame:
            return pd.DataFrame({"model": [], "fold": [], "group": []})
        def add_to_df(df: pd.DataFrame, datakey: DataKey, cols: Dict[str, List]) -> pd.DataFrame:
            data_len = len(next(iter(cols.values())))
            new_row_data = {"model": [datakey.feature_type] * data_len, "fold": [datakey.fold] * data_len, "group": [datakey.group] * data_len}
            new_row_data.update(cols)
            new_row = pd.DataFrame(new_row_data)
            return pd.concat([df, new_row], ignore_index=True)

        logging.info("Scoring models...")
        score_df = new_datakey_df()
        score_df["cindex"] = []

        if args.ensemble == "cat":
            # Relieve memory pressure by dropping training data
            del train_inputs
            del train_feature_data
            # Now, as late as possible, cat test data for evaluation
            cat_unimodal_data(test_feature_data)
            # for datakey, model in models.items():
            #     scores[datakey] = model.score(
            #         test_feature_data[datakey], test_outcomes[datakey]
            #     )
        elif args.ensemble in ["mean", "meanrank", "cox", "ipcr"]:
            def train_model(models: List[CoxPHSurvivalAnalysis], train_data: List[pd.DataFrame], train_outcomes: np.rec.recarray, args: Namespace) -> EnsembleModel:
                model = EnsembleModel(models, args)
                model.train(train_data, train_outcomes)
                return model

            # Generate all the ensemble keys we care about
            train_eks = ensemble_keys(train_groups, args)
            test_eks = ensemble_keys(test_groups, args)

            def expand_ensemble_key(ek: DataKey) -> List[DataKey]:
                return [DataKey(feature_type=ft, fold=ek.fold, group=ek.group) for ft in feature_set_from_label(ek.feature_type)]

            reducer_output = Parallel(
                return_as="generator", n_jobs=min(os.cpu_count() - 2, len(train_eks))
            )(
                delayed(train_model)(
                    [base_models[k] for k in expand_ensemble_key(ek)],
                    [train_feature_data[k] for k in expand_ensemble_key(ek)],
                    train_outcomes[ek],
                    args
                )
                for ek in train_eks
            )
            ensemble_models: Dict[DataKey, EnsembleModel] = {} if args.train_group_by else MappingDict(ignoring_group)
            for ek, ensemble_model in zip(train_eks, reducer_output, strict=True):
                ensemble_models[ek] = ensemble_model

            for ek in test_eks:
                if len(np.where(test_outcomes[ek]["death_witnessed"])[0]) < 2:
                    logging.info(f"Not enough deaths to score {ek}, skipping")
                    continue
                cindex = calculate_c_index(
                    args,
                    train_outcomes[ek],
                    test_outcomes[ek],
                    ensemble_models[ek].predict([test_feature_data[k] for k in expand_ensemble_key(ek)]),
                )  # TODO: consider using other c index outputs
                score_df = add_to_df(score_df, ek, {"cindex": [cindex]})
        else:
            # Unreachable
            assert False

        logging.info("Writing model scores...")
        with open(runpath("run.json"), "w+") as run_json:
            run = {}
            run["args"] = vars(args)
            json.dump(run, run_json)
        score_df.to_csv(runpath("scores.csv"), index=False)
        if args.vis_data:
            predicted_curves = new_datakey_df()
            predicted_curves["case"] = []
            predicted_curves["x"] = []
            predicted_curves["y"] = []
            predicted_curves["risk_score"] = []
            # Map of model type -> case -> survival curve predicted
            logging.info("Predicting survival curves for all patients in test sample...")
            # Our models cannot extrapolate beyond the furthest timepoint they've observed.
            # To make it simple, we only query models for time points that ALL of them can
            # extrapolate to. This drops evaluating survival at the distant tail, which is OK
            # since it probably works really badly there anyway.
            # Assuming random distribution of folds, this usually doesn't lose much data.
            unique_times_unfiltered = outcomes_df["days_to_event"].unique().tolist()
            unique_times = sorted(filter(lambda day: day <= model_extrapolation_limit, unique_times_unfiltered))
            logging.info(f"Extrapolation limit dropped {len(unique_times_unfiltered) - len(unique_times)} timepoints for visualization")
            for datakey in test_eks:
                model = ensemble_models[datakey]
                test_data: List[pd.DataFrame] = [test_feature_data[k] for k in expand_ensemble_key(datakey)]
                curves = model.predict_survival_function(test_data, unique_times)
                risk_scores = model.predict(test_data)
                data_len = len(curves)
                assert data_len == len(risk_scores)
                assert data_len == len(test_data[0].index)
                # All indexes should be the same
                predicted_curves = add_to_df(predicted_curves, datakey, {"case": list(test_data[0].index), "x": [unique_times] * data_len, "y": curves, "risk_score": risk_scores})
            logging.info(f"Writing predicted survival curves...")
            predicted_curves.to_csv(runpath("curves.csv"), index=False)
            logging.info(f"Written to curves.csv")
            logging.info(f"Writing outcomes file...")
            outcomes_df.to_csv(runpath("outcomes.csv"))
            logging.info(f"Written to outcomes.csv")
        logging.info(f"All files written, took {time.time() - start} seconds")
        logging.info(f"run path was {runpath('')}")

if __name__ == "__main__":
    main()
