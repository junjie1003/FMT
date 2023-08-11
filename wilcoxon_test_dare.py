import io
import sys
import argparse
import warnings
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank test")
    parser.add_argument("--testwith", type=str, choices=["CW", "FGSM", "JSMA", "PGD"])
    parser.add_argument("--model", type=str, choices=["VGG16", "VGG19", "Alexnet"])
    parser.add_argument("--dataset", type=str, choices=["CIFAR10", "SVHN", "FM"])
    args = parser.parse_args()

    return args


f = open("./log/wilcoxon_test_dare.log", "a", encoding="UTF-8")
sys.stdout = f
sys.stderr = f

args = get_args()
print(f"Model: {args.model}\nDataset: {args.dataset}\nTestwith: {args.testwith}\n")

path = "./data/dare.xlsx"
approaches = ["Dare", "CW", "FGSM", "JSMA", "PGD"]

if args.testwith == "CW":
    col_range = "C:G"
elif args.testwith == "FGSM":
    col_range = "H:L"
elif args.testwith == "JSMA":
    col_range = "M:Q"
elif args.testwith == "PGD":
    col_range = "R:V"

if args.model == "VGG16":
    base_row = 3
elif args.model == "VGG19":
    base_row = 30
elif args.model == "Alexnet":
    base_row = 57

if args.dataset == "CIFAR10":
    skip_row = base_row
elif args.dataset == "SVHN":
    skip_row = base_row + 9
elif args.dataset == "FM":
    skip_row = base_row + 18

df = pd.read_excel(path, sheet_name=0, header=None, usecols=col_range, skiprows=skip_row, nrows=9)
df.columns = approaches
print(df)

approaches.remove("Dare")

for app in approaches:
    print(f"\nDare v.s. {app}\n")
    w1, p1 = wilcoxon(df["Dare"].to_list(), df[app].to_list(), alternative="greater")
    print("Wilcoxon signed-rank test statistic:", w1)
    print("pvalue:", p1)
    alpha = 0.05
    if p1 < alpha:
        print("Rejecting the null hypothesis, your method exhibits a significantly smaller bias than the baseline methods.")
    else:
        print(
            "Failing to reject the null hypothesis, there is no significant difference in bias between your method and the baseline methods."
        )

    print(f"\n{app} v.s. Dare\n")
    w2, p2 = wilcoxon(df[app].to_list(), df["Dare"].to_list(), alternative="greater")
    print("Wilcoxon signed-rank test statistic:", w2)
    print("pvalue:", p2)
    alpha = 0.05
    if p2 < alpha:
        print("Rejecting the null hypothesis, your method exhibits a significantly smaller bias than the baseline methods.")
    else:
        print(
            "Failing to reject the null hypothesis, there is no significant difference in bias between your method and the baseline methods."
        )

print("-" * 120)

# if __name__ == "__main__":
#     w, p = wilcoxon([92.3, 91.4, 92.2, 91.8, 92.0, 91.5, 91.9, 92.3, 92.1], [75.5, 76.0, 75.4, 77.3, 74.9, 76.3, 77.8, 75.3, 75.4],
#                     alternative="greater")
#     print("Wilcoxon signed-rank test statistic:", w)
#     print("pvalue:", p)
#     alpha = 0.05
#     if p < alpha:
#         print("Rejecting the null hypothesis, your method exhibits a significantly smaller bias than the baseline methods.")
#     else:
#         print(
#             "Failing to reject the null hypothesis, there is no significant difference in bias between your method and the baseline methods."
#         )
