import os

ROOT_PATH = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(ROOT_PATH, "results")
DATA_PATH = os.path.join(ROOT_PATH, "TCGA-data")
CLINICAL_TSV_PATH = os.path.join(DATA_PATH, "clinical.tsv")


ELIGIBLE_MODEL_TYPES = ["cox", "ipcr", "rf", "gb", "est", "cgb"]
FEATURE_ID_DELIM = "-"
ELIGIBLE_FEATURE_TYPES = frozenset(
    [
        "project",
        "demo",
        "cancer",
        "expr",
        "rawexpr",
        "text",
        "text_sum_bm",
        "text_sum_bm_cor",
        "text_sum_m",
        "text_sum_co",
        "hist_mean",
        "hist_max",
    ]
)
# Thee feature types have duplicated data somehow and shouldn't be used together
# TODO: Maybe using them together wouldn't be a bad idea...? Something interesting to try
INCOMPATIBLE_FEATURE_TYPES = [
    frozenset(["expr", "rawexpr"]),
    frozenset(["hist_mean", "hist_max"]),
    frozenset(["text", "text_sum_bm", "text_sum_bm_cor", "text_sum_m", "text_sum_co"]),
]
for ft in ELIGIBLE_FEATURE_TYPES:
    assert FEATURE_ID_DELIM not in ft
for ift_set in INCOMPATIBLE_FEATURE_TYPES:
    assert ift_set.issubset(ELIGIBLE_FEATURE_TYPES)

CORRECTED_REPORTS = [
    "TCGA-22-A5C4",
    "TCGA-2Y-A9H9",
    "TCGA-3X-AAVA",
    "TCGA-85-7844",
    "TCGA-A8-A07F",
    "TCGA-AA-3680",
    "TCGA-AJ-A2QN",
    "TCGA-AK-3436",
    "TCGA-AN-A0XT",
    "TCGA-B1-A655",
    "TCGA-D7-A6F2",
    "TCGA-EB-A24C",
    "TCGA-FE-A23A",
    "TCGA-HT-7475",
    "TCGA-IP-7968",
    "TCGA-KK-A59V",
    "TCGA-P5-A730",
    "TCGA-QR-A6GR",
    "TCGA-S9-A6TS",
    "TCGA-SI-AA8C",
    "TCGA-EJ-A65F",
    "TCGA-DX-A6B9",
    "TCGA-LK-A4NW",
    "TCGA-OR-A5JM",
    "TCGA-A4-A48D",
    "TCGA-DX-A1KY",
    "TCGA-X2-A95T",
    "TCGA-A2-A25A",
    "TCGA-KK-A5A1",
    "TCGA-A3-A8OX",
    "TCGA-24-2035",
    "TCGA-66-2737",
    "TCGA-A8-A081",
    "TCGA-38-4629",
    "TCGA-DJ-A2PU",
    "TCGA-BJ-A0ZC",
    "TCGA-AC-A3YJ",
    "TCGA-AG-4021",
    "TCGA-50-5932",
    "TCGA-N8-A4PP",
]

CHANGED_CORRECTED_REPORTS = [
    "TCGA-2Y-A9H9",
    "TCGA-3X-AAVA",
    "TCGA-85-7844",
    "TCGA-AN-A0XT",
    "TCGA-EB-A24C",
    "TCGA-HT-7475",
    "TCGA-IP-7968",
    "TCGA-KK-A59V",
    "TCGA-EJ-A65F",
    "TCGA-LK-A4NW",
    "TCGA-DX-A1KY",
    "TCGA-X2-A95T",
    "TCGA-A2-A25A",
    "TCGA-KK-A5A1",
    "TCGA-A3-A8OX",
    "TCGA-24-2035",
    "TCGA-66-2737",
    "TCGA-A8-A081",
    "TCGA-38-4629",
    "TCGA-DJ-A2PU",
    "TCGA-AC-A3YJ",
    "TCGA-50-5932",
    "TCGA-N8-A4PP",
]
