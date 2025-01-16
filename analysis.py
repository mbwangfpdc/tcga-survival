#!/usr/bin/env python3

import statistics as stats
import json
import os
import sys
import argparse
from collections import defaultdict
from utils import *
from globals import ELIGIBLE_FEATURE_TYPES, OUTPUT_PATH
from statsmodels.stats.weightstats import ttost_paired
from statsmodels.stats.multicomp import MultiComparison

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("")
#     return parser.parse_args()

run = {}
with open(os.path.join(OUTPUT_PATH, sys.argv[1], "run.json")) as runjson:
    run = json.load(runjson)

results = {}
score_arr = []
score_labels = []
for fid, scores in run["scores"].items():
    # if "-" in fid:
    #     continue
    mean = stats.mean(scores)
    std = stats.stdev(scores)
    # print(f"{fid},{mean},{std}")
    results[fid] = (mean, std)
    for score in scores:
        score_labels.append(fid)
        score_arr.append(score)

multi = MultiComparison(score_arr, score_labels)
def func(x, y):
    res = ttost_paired(x, y, low=-0.005, upp=0.005)
    return (0, 1-res[0])
# print(multi.allpairtest(func)[0].as_csv())

eligible_feature_types = sorted(list(ELIGIBLE_FEATURE_TYPES))
def get_feature_types(fid: str) -> frozenset:
    feature_types = fid.split("-")
    return ",".join(map(lambda eft: str(eft in feature_types), eligible_feature_types))

sorted_results = sorted(results.items(), key=lambda item: item[1][0], reverse=True)
cols = eligible_feature_types.copy() + ["mean", "stddev"]
score_map: dict[frozenset, tuple[float, float]] = {}
print("===== MODEL SCORES =====")
for fid, (mean, std) in sorted_results:
    # if "rawexpr" in fid:
    #     continue
    if "demo" in fid and "-" in fid:
        continue
    score_map[frozenset(fid.split("-"))] = (mean, std)
    print(f"{fid},{mean:.5f},{std:.5f}")

fsets = get_feature_sets(ELIGIBLE_FEATURE_TYPES)

improvements = defaultdict(dict)
combos = []
for features, (mean, std) in score_map.items():
    for l, r in set_partitions(features):
        if score_map[l][0] < mean and score_map[r][0] < mean:
            if score_map[l][0] < score_map[r][0]:
                strong = r
                weak = l
            else:
                strong = l
                weak = r
            diff = mean - score_map[strong][0]
            diff_std = diff / std
            if diff_std > 1.5:
                combos.append((features, mean, std, diff, diff_std, strong, weak))
                # print(f"{feature_id(features)}:{mean:.5f} BETTER by {diff_std:.2f} stdev than components {feature_id(l)}:{score_map[l][0]:.5f}, {feature_id(r)}:{score_map[r][0]:.5f}")
                # print(f"{feature_id(features)}:{mean:.5f},{diff_std:.2f},{score_map[l][0]:.5f},{score_map[r][0]:.5f},{feature_id(l)},{feature_id(r)}")

sorted_combos = sorted(combos, key=lambda item: item[3], reverse=True)
print("===== PRODUCTIVE COMBOS =====")
for features, mean, std, diff, diff_std, strong, weak in sorted_combos:
    strong_score = score_map[strong][0]
    weak_score = score_map[weak][0]
    print(f"{feature_id(features)},{mean:.5f},{strong_score:.5f},{weak_score:.5f},{feature_id(strong)},{std:.5f},{diff_std:.5f}")

