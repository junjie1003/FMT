import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["FMT$^{-s}$", "FMT$^{-sl}$", "FMT$^{rand}$"]

single = [-0.0096, -0.0083, -0.0048]
dual = [-0.0033, 0.0003, -0.0034]

plt.figure(figsize=(20, 5))
plt.rc("font", family="Times New Roman", size=16)

plt.subplot(132)
x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.1  # 柱子的宽度

# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
plt.bar(x - width * 0.5, single, width, label="FMT$_s$", color="#5B9BD5")
plt.bar(x + width * 0.5, dual, width, label="FMT$_d$", color="#ED7D31")


# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)

y_major_locator = plt.MultipleLocator(0.01)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(-0.02, 0.02)

plt.grid(linestyle="--", alpha=0.3)
plt.legend()
plt.savefig(fname="../RQ7_results/ablation_accuracy.pdf", bbox_inches="tight")
