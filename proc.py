from numpy import mean

metrics = ["accuracy", "spd", "aaod", "eod"]
baselines = ["rew", "adv", "roc", "care", "supp"]
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


def table9():
    results = {}
    for baseline in baselines:
        results[baseline] = {}
        for metric in metrics:
            results[baseline][metric] = {}
            for task in tasks:
                results[baseline][metric][task] = {}

    for baseline in baselines:
        for metric in metrics:
            for task in tasks:
                for method in methods:
                    base_accuracy, base_spd, base_aaod, base_eod = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
                    accuracy, spd, aaod, eod = get_metrics(f"./RQ5_results/fmt/fmt_{method}_{task}.txt")
                    results[baseline]["accuracy"][task][method] = (accuracy-base_accuracy) * 100
                    results[baseline]["spd"][task][method] = (base_spd-spd) * 100
                    results[baseline]["aaod"][task][method] = (base_aaod-aaod) * 100
                    results[baseline]["eod"][task][method] = (base_eod-eod) * 100

    for baseline in baselines:
        for metric in metrics:
            print("")
            print(baseline, metric)
            print("")
            for task in tasks:
                for method in methods:
                    print(task, method)
                    print("%.2f" % (results[baseline][metric][task][method]))


def fmt_s_aaod():
    results = []
    for baseline in baselines:
        for task in tasks:
            _, _, base_aaod, _ = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
            _, _, aaod, _ = get_metrics(f"./RQ5_results/fmt/fmt_single_{task}.txt")
            results.append((base_aaod-aaod) / base_aaod)

    print("%.2f" % (mean(results) * 100))


def fmt_s_eod():
    results = []
    for baseline in baselines:
        for task in tasks:
            _, _, _, base_eod = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
            _, _, _, eod = get_metrics(f"./RQ5_results/fmt/fmt_single_{task}.txt")
            results.append((base_eod-eod) / base_eod)

    print("%.2f" % (mean(results) * 100))


def fmt_d_aaod():
    results = []
    for baseline in baselines:
        for task in tasks:
            _, _, base_aaod, _ = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
            _, _, aaod, _ = get_metrics(f"./RQ5_results/fmt/fmt_dual_{task}.txt")
            results.append((base_aaod-aaod) / base_aaod)

    print("%.2f" % (mean(results) * 100))


def fmt_d_eod():
    results = []
    for baseline in baselines:
        for task in tasks:
            _, _, _, base_eod = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
            _, _, _, eod = get_metrics(f"./RQ5_results/fmt/fmt_dual_{task}.txt")
            results.append((base_eod-eod) / base_eod)

    print("%.2f" % (mean(results) * 100))


def fmt_s_acc():
    results = []
    for baseline in baselines:
        for task in tasks:
            base_acc, _, _, _ = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
            acc, _, _, _ = get_metrics(f"./RQ5_results/fmt/fmt_single_{task}.txt")
            results.append((acc-base_acc) / base_acc)

    print("%.2f" % (mean(results) * 100))


def fmt_d_acc():
    results = []
    for baseline in baselines:
        for task in tasks:
            base_acc, _, _, _ = get_metrics(f"./RQ5_results/{baseline}/{baseline}_{task}.txt")
            acc, _, _, _ = get_metrics(f"./RQ5_results/fmt/fmt_dual_{task}.txt")
            results.append((acc-base_acc) / base_acc)

    print("%.2f" % (mean(results) * 100))


if __name__ == "__main__":
    # fmt_s_aaod()
    # fmt_s_eod()
    # fmt_d_aaod()
    # fmt_d_eod()
    fmt_s_acc()
    fmt_d_acc()
