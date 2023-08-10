import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def get_metric_results(file_path, metric):
    with open(file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            if metric in line and "change" not in line:
                results = [float(res) for res in line.strip().split("\t")[1:11]]

    return results


# metric = "accuracy"
# metric = "spd"
# metric = "aaod"
metric = "eod"
augmentation = ["single", "dual"]
approaches = ["FMT", "REW", "ADV", "ROC", "CARE", "SUPP"]
tasks = ["adult_race", "adult_sex", "bank_age", "compas_race", "compas_sex", "german_age", "german_sex"]

# # All subjects
# result_dict = {}
# for app in approaches:
#     result_dict[app] = {}
#     for aug in augmentation:
#         result = []
#         for task in tasks:
#             if app == "FMT":
#                 file_path = f"./RQ5_results/{app.lower()}/{app.lower()}_{aug}_{task}.txt"
#             elif app == "SUPP":
#                 file_path = f"./Major/suppression/suppression_{task}.txt"
#             else:
#                 file_path = f"./RQ5_results/{app.lower()}/{app.lower()}_{task}.txt"

#             temp = get_metric_results(file_path, metric)
#             result.extend(temp)
#         result_dict[app][aug] = result

# for app in ["REW", "ADV", "ROC", "CARE", "SUPP"]:
#     for aug in augmentation:
#         print(f"Ours v.s. {app} ({aug})")
#         w, p = wilcoxon(result_dict["FMT"][aug], result_dict[app][aug], alternative="greater")
#         # w, p = wilcoxon(result_dict["FMT"][aug], result_dict[app][aug], alternative="less")
#         print("Wilcoxon Test统计量:", w)
#         print("单侧P值:", p)
#         alpha = 0.05
#         if p < alpha:
#             print("拒绝原假设，您的方法的偏差显著小于基线方法的偏差。")
#         else:
#             print("不能拒绝原假设，您的方法的偏差与基线方法的偏差无显著差异。")

# Each subject
result_dict = {}
for app in approaches:
    result_dict[app] = {}
    for task in tasks:
        result_dict[app][task] = {}
        for aug in augmentation:
            result = []
            if app == "FMT":
                file_path = f"./RQ5_results/{app.lower()}/{app.lower()}_{aug}_{task}.txt"
            elif app == "SUPP":
                file_path = f"./Major/suppression/suppression_{task}.txt"
            else:
                file_path = f"./RQ5_results/{app.lower()}/{app.lower()}_{task}.txt"

            temp = get_metric_results(file_path, metric)
            result.extend(temp)
            result_dict[app][task][aug] = result

for app in ["REW", "ADV", "ROC", "CARE", "SUPP"]:
    for task in tasks:
        for aug in augmentation:
            print(f"Ours v.s. {app} ({task} {aug})")
            # w1, p1 = wilcoxon(result_dict["FMT"][task][aug], result_dict[app][task][aug], alternative="greater")
            w1, p1 = wilcoxon(result_dict["FMT"][task][aug], result_dict[app][task][aug], alternative="less")
            print("Wilcoxon Test统计量:", w1)
            print("单侧P值:", p1)
            alpha = 0.05
            if p1 < alpha:
                print("拒绝原假设，您的方法的偏差显著小于基线方法的偏差。")
            else:
                print("不能拒绝原假设，您的方法的偏差与基线方法的偏差无显著差异。")

            # w2, p2 = wilcoxon(result_dict[app][task][aug], result_dict["FMT"][task][aug], alternative="greater")
            w2, p2 = wilcoxon(result_dict[app][task][aug], result_dict["FMT"][task][aug], alternative="less")
            print("Wilcoxon Test统计量:", w2)
            print("单侧P值:", p2)
            alpha = 0.05
            if p2 < alpha:
                print("拒绝原假设，您的方法的偏差显著小于基线方法的偏差。")
            else:
                print("不能拒绝原假设，您的方法的偏差与基线方法的偏差无显著差异。")
