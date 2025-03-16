#!/usr/bin/env python3

import statistics as stats
import json
import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from utils import *
from globals import ELIGIBLE_FEATURE_TYPES, OUTPUT_PATH
from statsmodels.stats.weightstats import ttost_paired
from statsmodels.stats.multicomp import MultiComparison
from matplotlib import pyplot as plt

PCAS = [4, 8, 16, 32, 64, 128, 256, 512]
ENSEMBLES = ["cox", "mean"]
RUNS = [(npca, ensemble) for npca in PCAS for ensemble in ENSEMBLES]

# TODO: Drop models with _cor, pick a hist, maybe ditch expr
def make_table():
    for ensemble in ENSEMBLES:
        df = pd.DataFrame()
        for npca in PCAS:
            model_results = get_model_results(f"{npca}cox{ensemble}").rename(columns={'mean': npca}).drop(columns="std")
            df = df.join(model_results, how="outer")
        df.to_csv(f"scores_{ensemble}.csv")
        # TODO: remove this when we generate full table of data
        df = df[~df.index.str.contains("text_sum_bm_cor|text_sum_co|text_sum_m|hist_mean")]
        # Use a synthetic column to sort
        # Sort models by the peak performance of any PCA size
        df["sort_key"] = df.apply(lambda row: row.max(), axis=1)
        df.sort_values("sort_key", ascending=False, inplace=True)
        df.drop(columns="sort_key", inplace=True)
        print(f"DF for {ensemble}")
        print(df)
        df.to_html(f"temp-{ensemble}")
        column_names = tuple(df.columns)
        row_names = list(df.index)
        cell_text = []
        # colors = []
        normalized = (df.to_numpy() - 0.5) * 2
        colors = plt.cm.BuPu(normalized)
        for row in df.to_numpy():
            cell_text.append([])
            # colors.append([])
            for val in row:
                cell_text[-1].append(f"{val:.4f}")

        # plt.table(cellText=df.to_numpy().astype('|S6'), rowLabels=row_names, colLabels=column_names, loc="center")
        # plt.axis('off')
        # plt.show()


        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))  # Set the figure size

        # Hide the axes
        ax.axis('off')

        # Create the table and stretch it to fit the figure
        ax.table(cellText=cell_text, rowLabels=row_names, colLabels=column_names, cellColours=colors, loc="center")

        # Adjust layout to make the table fit the entire figure
        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
        plt.tight_layout()

        # Display the table
        plt.show()
        plt.savefig(f"temp-{ensemble}")

def group_scores(group) -> pd.DataFrame:
    return pd.DataFrame({"mean": [group["cindex"].mean()], "std": [group["cindex"].std()]})

def get_model_results(run_dir: str) -> pd.DataFrame:
    score_df = pd.read_csv(os.path.join(OUTPUT_PATH, run_dir, "scores.csv"))
    return score_df.groupby("model")["cindex"].agg(["mean", "std"])

def print_analysis(run_dir: str):
    results = get_model_results(run_dir)

    # def func(x, y):
    #     res = ttost_paired(x, y, low=-0.005, upp=0.005)
    #     return (0, 1-res[0])
    # print(multi.allpairtest(func)[0].as_csv())

    eligible_feature_types = sorted(list(ELIGIBLE_FEATURE_TYPES))

    sorted_results = sorted(results.items(), key=lambda item: item[1][0], reverse=True)
    score_map: dict[frozenset, tuple[float, float]] = {}
    print("===== MODEL SCORES =====")
    for fid, (mean, std) in sorted_results:
        #     continue
        if "demo" in fid and FEATURE_ID_DELIM in fid:
            continue
        score_map[frozenset(feature_set_from_label(fid))] = (mean, std)
        print(f"{fid},{mean:.5f},{std:.5f}")

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
        print(f"{feature_set_label(features)},{mean:.5f},{strong_score:.5f},{weak_score:.5f},{feature_set_label(strong)},{std:.5f},{diff_std:.5f}")

if __name__ == "__main__":
    make_table()
