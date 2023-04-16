import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["SPD", "AAOD", "EOD"]

single_s = [0.1466, -0.6773, -0.1188]
dual_s = [0.0710, -0.5114, -1.0194]

single_sl = [0.1302, -0.5088, -0.3900]
dual_sl = [-0.0370, -0.1611, -0.9927]

single_last = [-0.0555, -0.2118, -0.4323]
dual_last = [-0.0606, -0.0493, -0.3734]

single_rand = [-0.1836, -1.3843, -4.7567]
dual_rand = [-0.3258, -0.9484, -3.1847]

plt.figure(figsize=(45, 6))
plt.rc("font", family="Times New Roman", size=16)

plt.subplot(143)
x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.1  # 柱子的宽度

# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
plt.bar(x - width * 3.5, single_s, width, label="FMT$_s^{-s}$", color="#5B9BD5")
plt.bar(x - width * 2.5, dual_s, width, label="FMT$_d^{-s}$", color="#ED7D31")
plt.bar(x - width * 1.5, single_sl, width, label="FMT$_s^{-sl}$", color="#A5A5A5")
plt.bar(x - width * 0.5, dual_sl, width, label="FMT$_s^{-sl}$", color="#B8BBDE")
plt.bar(x + width * 0.5, single_last, width, label="FMT$_s^{last}$", color="#FFC000")
plt.bar(x + width * 1.5, dual_last, width, label="FMT$_d^{last}$", color="#4472C4")
plt.bar(x + width * 2.5, single_rand, width, label="FMT$_s^{rand}$", color="#70AD47")
plt.bar(x + width * 3.5, dual_rand, width, label="FMT$_d^{rand}$", color="#F2B0AC")

# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)

y_major_locator = plt.MultipleLocator(0.50)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(-5, 1)

plt.grid(linestyle="--", alpha=0.3)
plt.legend(loc="lower left")
plt.savefig(fname="../RQ6&7_results/ablation_fairness.pdf", bbox_inches="tight")
