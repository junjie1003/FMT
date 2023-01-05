import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["SPD", "AAOD", "EOD"]
single_s = [0.1466, -0.6773, -0.1188]
dual_s = [0.0710, -0.5114, -1.0194]
single_sl = [0.1302, -0.5088, -0.3900]
dual_sl = [-0.0370, -0.1611, -0.9927]
single_rand = [-0.1836, -1.3843, -4.7567]
dual_rand = [-0.3258, -0.9484, -3.1847]

# single_s = [14.66, -67.73, -11.88]
# dual_s = [7.10, -51.14, -101.94]
# single_sl = [13.02, -50.88, -39.00]
# dual_sl = [-3.70, -16.11, -99.27]
# single_rand = [-18.36, -138.43, -475.67]
# dual_rand = [-32.58, -94.84, -318.47]

plt.figure(figsize=(30, 5))
plt.rc("font", family="Times New Roman", size=16)

plt.subplot(132)
x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.1  # 柱子的宽度

# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
plt.bar(x - width * 2.5, single_s, width, label="FMT$_s^{-s}$", color="#5B9BD5")
plt.bar(x - width * 1.5, dual_s, width, label="FMT$_d^{-s}$", color="#ED7D31")
plt.bar(x - width * 0.5, single_sl, width, label="FMT$_s^{-sl}$", color="#A5A5A5")
plt.bar(x + width * 0.5, dual_sl, width, label="FMT$_d^{-sl}$", color="#FFC000")
plt.bar(x + width * 1.5, single_rand, width, label="FMT$_s^{rand}$", color="#4472C4")
plt.bar(x + width * 2.5, dual_rand, width, label="FMT$_d^{rand}$", color="#70AD47")

# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)

y_major_locator = plt.MultipleLocator(0.50)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(-5, 1)

plt.grid(linestyle="--", alpha=0.3)
plt.legend()
plt.savefig(fname="../RQ7_results/ablation_fairness.pdf", bbox_inches="tight")
