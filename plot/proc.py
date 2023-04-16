from numpy import mean

metrics = ["accuracy", "spd", "aaod", "eod"]
# variants = ["fmt_remove_s", "fmt_remove_sl", "fmt_last", "fmt_rand"]
baselines = ["rew", "adv", "roc", "care"]
methods = ["single", "dual"]
tasks = ["adult_race", "adult_sex", "bank_age", "compas_race", "compas_sex", "german_age", "german_sex"]


def get_metrics(filename):
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            if "accuracy" in line and "change" not in line:
                accuracy = float(line.strip().split("\t")[-2])
            if "spd" in line and "change" not in line:
                spd = float(line.strip().split("\t")[-2])
            if "aaod" in line and "change" not in line:
                aaod = float(line.strip().split("\t")[-2])
            if "eod" in line and "change" not in line:
                eod = float(line.strip().split("\t")[-2])

    return accuracy, spd, aaod, eod


def get_metric_changes(filename):
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            if "accuracy change:" in line:
                accuracy = float(line.strip().split("accuracy change: ")[1])
            if "spd change:" in line:
                spd = float(line.strip().split("spd change: ")[1])
            if "aaod change:" in line:
                aaod = float(line.strip().split("aaod change: ")[1])
            if "eod change:" in line:
                eod = float(line.strip().split("eod change: ")[1])

    return accuracy, spd, aaod, eod


# results = {}
# for metric in metrics:
#     results[metric] = {}
#     for variant in variants:
#         results[metric][variant] = {}
#         for method in methods:
#             results[metric][variant][method] = []


# for metric in metrics:
#     for variant in variants:
#         for method in methods:
#             for task in tasks:
#                 base_accuracy, base_spd, base_aaod, base_eod = get_metrics("./RQ5_results/fmt/fmt_%s_%s.txt" % (method, task))
#                 accuracy, spd, aaod, eod = get_metrics(
#                     "./RQ6&7_results/%s/%s_%s_%s.txt" % (variant.split("fmt_")[1], variant, method, task)
#                 )
#                 results["accuracy"][variant][method].append((accuracy - base_accuracy) / base_accuracy)
#                 results["spd"][variant][method].append((base_spd - spd) / base_spd)
#                 results["aaod"][variant][method].append((base_aaod - aaod) / base_aaod)
#                 results["eod"][variant][method].append((base_eod - eod) / base_eod)

results = {}
for metric in metrics:
    results[metric] = {}
    for method in methods:
        results[metric][method] = []

for metric in metrics:
    for method in methods:
        for baseline in baselines:
            for task in tasks:
                base_accuracy, base_spd, base_aaod, base_eod = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
                accuracy, spd, aaod, eod = get_metrics(f"./RQ5_results/fmt/fmt_{method}_{task}.txt")
                results["accuracy"][method].append((accuracy - base_accuracy) / base_accuracy)
                results["spd"][method].append((base_spd - spd) / base_spd)
                results["aaod"][method].append((base_aaod - aaod) / base_aaod)
                results["eod"][method].append((base_eod - eod) / base_eod)


for metric in metrics:
    for method in methods:
        print(metric, method)
        print("%.4f" % (mean(results[metric][method])))
