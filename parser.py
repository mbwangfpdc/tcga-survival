import argparse

from globals import ELIGIBLE_FEATURE_TYPES, ELIGIBLE_MODEL_TYPES, FEATURE_ID_DELIM
from utils import get_timestamp, get_feature_sets, feature_set_label, feature_set_from_label

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
        "--ensembles",
        nargs="*",
        help=f"Which ensembles to run. Unimodal models that must be trained are inferred from the ensembles. For example, '--ensembles expr-text text-rawexpr' will train expr, text, and rawexpr, then evaluate the two provided ensembles. Cannot be provided alongside --features or --solo_features.",
        default=[],
    )
    parser.add_argument(
        "--features",
        nargs="*",
        help=f"Which features to train models over. All possible combinations of features specified here are run. Feature sets are concatenated by '{FEATURE_ID_DELIM}' in alphabetical order. If 'all' is contained in the input, run all possible features. Cannot be provided alongside --ensembles.",
        default=[],
    )
    parser.add_argument(
        "--solo_features",
        nargs="*",
        help=f"Which features to run solo. Any features specified here will be run as a unimodal model. If this overlaps with features specified in --features, this wins (overlaped features are run like solo features). If 'all' is contained in the input, run all possible features. Cannot be provided alongside --ensembles.",
        default=[],
    )
    parser.add_argument(
        "--train_group_by",
        type=str,
        help=f"Which column (from the clinical data) to group by when training models. For example, project_id would train models specialized on each TCGA project. Specifying this without --test_group_by would result in evaluating specialized models on more general data. Currently breaks when run on small groups due to limitations of PCA",
        default=None,
    )
    parser.add_argument(
        "--test_group_by",
        type=str,
        help=f"Which column (from the clinical data) to group by when testing models. For example, project_id would report model performance on each TCGA project. Specifying this without --train_group_by would result in evaluating general models on more specialized data.",
        default=None,
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
        default="mean",
        choices=["cat", "mean", "meanrank", "cox", "ipcr"],
    )
    parser.add_argument(
        "--vis_data",
        action="store_true",
        help="Whether to emit data for visualizations. Takes a long time.",
        default=False,
    )
    parser.add_argument(
        "--cindex",
        type=str,
        help="Which measure of survival to use",
        default="harrel",
        choices=["harrel", "uno"],
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
        "--alpha", type=float, help="alpha value for Cox training", default=0.01
    )
    parser.add_argument(
        "--rand_state",
        type=int,
        help="Random state",
        default=42,
    )
    parser.add_argument(
        "--rundir",
        type=str,
        help="Name of subdirectory of ./results to dump results to, or 'dryrun' to emit no files.",
        default=get_timestamp(),
    )
    args = parser.parse_args()

    # Generate ensembles and unimodal models to train
    feature_sets = []
    if args.ensembles:
        if bool(args.features) or bool(args.solo_features):
            print("ERROR: cannot provide both ensembles and features.")
            exit(0)
        args.features = set()
        for ensemble in args.ensembles:
            feature_set = feature_set_from_label(ensemble)
            feature_sets.append(feature_set)
            for feature in feature_set:
                args.features.add(feature)
    else:
        if "all" in args.solo_features:
            args.solo_features = ELIGIBLE_FEATURE_TYPES
        if "all" in args.features:
            args.features = ELIGIBLE_FEATURE_TYPES
        # Solo features should not be combined, remove them
        args.features = list(set(args.features).difference(set(args.solo_features)))
        args.solo_features = list(args.solo_features)
        feature_sets = get_feature_sets(args.features) + [
            set([f]) for f in args.solo_features
        ]
        args.features = args.features + args.solo_features
    feature_sets.sort(key=feature_set_label)
    # These should be a serializable type like list
    args.ensembles = [list(fs) for fs in feature_sets]
    args.features = list(args.features)

    for feature_list in [args.features, args.solo_features]:
        for feature in feature_list:
            if feature not in ELIGIBLE_FEATURE_TYPES:
                print(f"{feature} not an eligible feature type")
                exit(1)

    if args.ensemble == "cat":
        print("SORRY, CAT DOESNT WORK RIGHT NOW, LUV U BYE")
        exit(1)
    return args
