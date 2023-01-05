from numpy import mean

metrics = ["accuracy", "spd", "aaod", "eod"]
variants = ["fmt_remove_s", "fmt_remove_st", "fmt_rand"]
methods = ["single", "dual"]
tasks = ["adult_race", "adult_sex", "bank_age", "compas_race", "compas_sex", "german_age", "german_sex"]
parameters = {
    "dual_adult_race": "0.001_32",
    "dual_adult_sex": "0.001_16",
    "dual_bank_age": "1e-05_128",
    "dual_compas_race": "0.001_16",
    "dual_compas_sex": "1e-05_64",
    "dual_german_age": "0.001_128",
    "dual_german_sex": "0.0001_16",
    "single_adult_race": "1e-05_32",
    "single_adult_sex": "1e-05_128",
    "single_bank_age": "1e-05_32",
    "single_compas_race": "0.0001_128",
    "single_compas_sex": "1e-05_16",
    "single_german_age": "0.001_128",
    "single_german_sex": "0.001_64",
}


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

    print(accuracy, spd, aaod, eod)

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


results = {}
for metric in metrics:
    results[metric] = {}
    for variant in variants:
        results[metric][variant] = {}
        for method in methods:
            results[metric][variant][method] = []


for metric in metrics:
    for variant in variants:
        for method in methods:
            for task in tasks:
                print(metric, variant, method, task)
                base_accuracy, base_spd, base_aaod, base_eod = get_metrics(
                    "./RQ5_results/fmt/fmt_%s_%s_%s.txt" % (method, task, parameters[method + "_" + task])
                )
                accuracy, spd, aaod, eod = get_metrics(
                    "./RQ6_results/%s/%s_%s_%s_%s.txt"
                    % (variant.split("fmt_")[1], variant, method, task, parameters[method + "_" + task])
                )

                results["accuracy"][variant][method].append((accuracy - base_accuracy) / base_accuracy)
                results["spd"][variant][method].append((base_spd - spd) / base_spd)
                results["aaod"][variant][method].append((base_aaod - aaod) / base_aaod)
                results["eod"][variant][method].append((base_eod - eod) / base_eod)


for metric in metrics:
    for variant in variants:
        for method in methods:
            print(metric, variant, method)
            print("%.4f" % (mean(results[metric][variant][method])))
