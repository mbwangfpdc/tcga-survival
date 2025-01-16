import argparse

from globals import ELIGIBLE_FEATURE_TYPES, ELIGIBLE_MODEL_TYPES, FEATURE_ID_DELIM
from utils import get_timestamp

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Cox Proportional Hazards model over TCGA data"
    )

    parser.add_argument(
        "--model",
        help=f"Which model to train",
        choices=ELIGIBLE_MODEL_TYPES,
        default="cox",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        help=f"Which featuress to run. All possible combinations of features specified here are run. Feature sets are concatenated by '{FEATURE_ID_DELIM}' in alphabetical order. If 'all' is contained in the input, run all possible features.",
        default=[],
    )
    parser.add_argument(
        "--solo_features",
        nargs="*",
        help=f"Which features to run solo. Any features specified here will be run as a unimodal model. If this overlaps with features specified in --features, this wins (overlaped features are run like solo features). If 'all' is contained in the input, run all possible features.",
        default=[],
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="Number of folds to cross-validate with. 5 or 10 is generally enough.",
        default=5,
    )
    parser.add_argument(
        "--pca_pre_join",
        type=int,
        help="We reduce feature vectors to this many components before concatenating them. If non-positive, not run.",
        default=0,
    )
    parser.add_argument(
        "--pca_post_join",
        type=int,
        help="We reduce post-concatenation feature vectors to this many components. If non-positive, not run.",
        default=0,
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        help="The way trained models are ensembled",
        default=["cat"],
        choices=["cat", "mean", "meanrank", "cox", "ipcr"],
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Max iterations to wait for a model to converge",
        default=100,
    )
    parser.add_argument(
        "--n_est",
        type=int,
        help="Number of estimators to use for ensemble model",
        default=100,
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        default=False,
        help="Save the models to the rundir. Adding this emits lots of data and takes extra time.",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        help="Chart the predicted survival curves over this many time steps",
        default=100,
    )
    parser.add_argument(
        "--alpha", type=float, help="alpha value for Cox training", default=0.01
    )
    parser.add_argument(
        "--rand_state",
        type=int,
        help="Random state to initialize fold distribution",
        default=42,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "analyze", "all"],
        default="train",
        help="The mode to run in. 'train' trains models and dumps their weights to the filesystem. 'analyze' intakes dumped models and outputs stats and figures. 'both' does both.",
    )
    parser.add_argument(
        "--rundir",
        type=str,
        help="Name of subdirectory of ./results to dump results to.",
        default=get_timestamp(),
    )
    args = parser.parse_args()
    if "all" in args.solo_features:
        args.solo_features = ELIGIBLE_FEATURE_TYPES
    if "all" in args.features:
        args.features = ELIGIBLE_FEATURE_TYPES
    # Solo features should not be combined, remove them
    args.features = set(args.features).difference(set(args.solo_features))
    # These should be a serializable type like list
    args.features = list(args.features)
    args.solo_features = list(args.solo_features)
    for feature_list in [args.features, args.solo_features]:
        for feature in feature_list:
            if feature not in ELIGIBLE_FEATURE_TYPES:
                print(f"{feature} not an eligible feature type")
                exit(1)
    return args
