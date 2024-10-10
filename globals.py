import os
import datetime

ROOT_PATH = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(ROOT_PATH, "results")


# Make a new unique results directory in results, then return the path
def make_results_dir() -> str:
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d-%H:%M:%S")
    new_results_dir = os.path.join(OUTPUT_PATH, formatted_datetime)
    os.makedirs(new_results_dir)
    return new_results_dir


DATA_PATH = os.path.join(ROOT_PATH, "TCGA-data")
CLINICAL_TSV_PATH = os.path.join(DATA_PATH, "clinical.tsv")


def feature_path_for(feature_type: str) -> str:
    return os.path.join(DATA_PATH, f"X_{feature_type}.h5ad")


ELIGIBLE_FEATURE_TYPES = ["expr", "text", "hist_mean", "hist_max"]
