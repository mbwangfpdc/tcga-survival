import os
import datetime

ROOT_PATH = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(ROOT_PATH, "results")
DATA_PATH = os.path.join(ROOT_PATH, "TCGA-data")
CLINICAL_TSV_PATH = os.path.join(DATA_PATH, "clinical.tsv")


ELIGIBLE_MODEL_TYPES = ["cox", "ipcr", "rf", "gb", "est", "cgb"]
FEATURE_ID_DELIM = "-"
ELIGIBLE_FEATURE_TYPES = frozenset(["project", "demo", "cancer", "expr", "rawexpr", "text", "text_sum_bm", "text_sum_co", "hist_mean", "hist_max"])
# Thee feature types have duplicated data somehow and shouldn't be used together
# TODO: Maybe using them together wouldn't be a bad idea...? Something interesting to try
INCOMPATIBLE_FEATURE_TYPES = [
    frozenset(["expr", "rawexpr"]),
    frozenset(["hist_mean", "hist_max"]),
    frozenset(["text", "text_sum_bm", "text_sum_co"]),
]
for ft in ELIGIBLE_FEATURE_TYPES:
    assert FEATURE_ID_DELIM not in ft
for ift_set in INCOMPATIBLE_FEATURE_TYPES:
    assert ift_set.issubset(ELIGIBLE_FEATURE_TYPES)
