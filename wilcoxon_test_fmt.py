import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
from openpyxl import load_workbook


def get_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--table-range", type=str, help="Table range")
    args = parser.parse_args()

    return args


def get_metric_results(file_path, metric):
    with open(file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            if metric in line and "change" not in line:
                results = [float(res) for res in line.strip().split("\t")[1:11]]

    return results


# metric = "accuracy"
metric = "eod"
augmentation = ["single", "dual"]
approaches = ["FMT", "REW", "ADV", "ROC", "CARE"]
tasks = ["adult_race", "adult_sex", "bank_age", "compas_race", "compas_sex", "german_age", "german_sex"]

result_dict = {}
for app in approaches:
    result_dict[app] = {}
    for aug in augmentation:
        result = []
        for task in tasks:
            if app == "FMT":
                file_path = f"./RQ5_results/{app.lower()}/{app.lower()}_{aug}_{task}.txt"
            else:
                file_path = f"./RQ5_results/{app.lower()}/{app.lower()}_{task}.txt"

            temp = get_metric_results(file_path, metric)
            result.extend(temp)
        result_dict[app][aug] = result

for app in ["REW", "ADV", "ROC", "CARE"]:
    for aug in augmentation:
        print(f"Ours v.s. {app} ({aug})")
        w, p = wilcoxon(result_dict["FMT"][aug], result_dict[app][aug], alternative="less")
        print("Wilcoxon Test统计量:", w)
        print("单侧P值:", p)
        alpha = 0.05
        if p < alpha:
            print("拒绝原假设，您的方法的偏差显著小于基线方法的偏差。")
        else:
            print("不能拒绝原假设，您的方法的偏差与基线方法的偏差无显著差异。")
