import argparse
import matplotlib.pyplot as plt


def data_helper(dataset, protected):
    task = dataset.capitalize() + "-" + protected.capitalize()
    print(task)

    if task == "Adult-Race":
        data = [
            [0.845065, 0.848972, 0.841601, 0.842854, 0.845286, 0.847719, 0.842707, 0.837252, 0.841085, 0.841822],  # REW
            [0.829439, 0.828333, 0.828849, 0.827228, 0.828481, 0.819931, 0.827744, 0.831577, 0.830840, 0.826270],  # ADV
            [0.845673, 0.834711, 0.837926, 0.845252, 0.835494, 0.844988, 0.838705, 0.844982, 0.842261, 0.846098],  # CARE
            [0.848456, 0.848530, 0.848382, 0.848456, 0.848456, 0.848603, 0.848530, 0.848530, 0.848530, 0.848456],  # FMT_s
            [0.843591, 0.842485, 0.843444, 0.843223, 0.842633, 0.843665, 0.844697, 0.844033, 0.844328, 0.843812],  # FMT_d
        ]

    elif task == "Adult-Sex":
        data = [
            [0.841969, 0.839095, 0.844844, 0.843223, 0.841822, 0.839906, 0.839685, 0.842043, 0.841822, 0.841380],  # REW
            [0.842780, 0.836515, 0.850299, 0.844107, 0.841011, 0.842191, 0.842707, 0.844033, 0.842191, 0.847055],  # ADV
            [0.842595, 0.831240, 0.843429, 0.843038, 0.842593, 0.846975, 0.844721, 0.838463, 0.834661, 0.842155],  # CARE
            [0.846392, 0.846466, 0.846466, 0.846539, 0.846466, 0.846318, 0.846466, 0.846392, 0.846539, 0.846392],  # FMT_s
            [0.844107, 0.842928, 0.843886, 0.844033, 0.843665, 0.843444, 0.843370, 0.843149, 0.843517, 0.844107],  # FMT_d
        ]

    elif task == "Bank-Age":
        data = [
            [0.890893, 0.890565, 0.889253, 0.888488, 0.884662, 0.885099, 0.882584, 0.886302, 0.880179, 0.894173],  # REW
            [0.894829, 0.895704, 0.895266, 0.901388, 0.896359, 0.898765, 0.898109, 0.895922, 0.895376, 0.896797],  # ADV
            [0.862392, 0.883472, 0.882104, 0.874342, 0.883073, 0.866629, 0.883514, 0.870286, 0.878150, 0.877604],  # CARE
            [0.896359, 0.896469, 0.896359, 0.896359, 0.896359, 0.896359, 0.896578, 0.896469, 0.896359, 0.896469],  # FMT_s
            [0.895376, 0.895813, 0.895813, 0.895376, 0.895704, 0.895485, 0.895485, 0.895485, 0.895813, 0.895704],  # FMT_d
        ]

    elif task == "Compas-Race":
        data = [
            [0.639114, 0.637493, 0.652620, 0.649919, 0.642896, 0.643436, 0.649919, 0.640735, 0.647218, 0.656402],  # REW
            [0.650999, 0.651540, 0.652080, 0.643436, 0.646677, 0.647218, 0.650999, 0.653160, 0.649919, 0.656942],  # ADV
            [0.647672, 0.646877, 0.638792, 0.641416, 0.638924, 0.642295, 0.641057, 0.644672, 0.645573, 0.639657],  # CARE
            [0.656402, 0.656402, 0.655862, 0.656402, 0.656402, 0.656402, 0.655862, 0.657482, 0.656942, 0.656942],  # FMT_s
            [0.644516, 0.642896, 0.654241, 0.648838, 0.657482, 0.640194, 0.646137, 0.654241, 0.651540, 0.649919],  # FMT_d
        ]

    elif task == "Compas-Sex":
        data = [
            [0.653701, 0.659103, 0.656942, 0.655321, 0.657482, 0.656942, 0.655321, 0.647758, 0.649919, 0.658563],  # REW
            [0.647758, 0.652080, 0.652080, 0.649379, 0.651540, 0.649379, 0.647218, 0.650999, 0.638033, 0.652080],  # ADV
            [0.646783, 0.650372, 0.639802, 0.642669, 0.646665, 0.641221, 0.647254, 0.639980, 0.640147, 0.645463],  # CARE
            [0.647218, 0.646677, 0.647218, 0.645057, 0.646677, 0.646137, 0.645057, 0.645597, 0.645597, 0.645057],  # FMT_s
            [0.647758, 0.647758, 0.648298, 0.647218, 0.647758, 0.646677, 0.646677, 0.648298, 0.647758, 0.647218],  # FMT_d
        ]

    elif task == "German-Age":
        data = [
            [0.696667, 0.766667, 0.690000, 0.750000, 0.743333, 0.730000, 0.750000, 0.750000, 0.750000, 0.750000],  # REW
            [0.740000, 0.740000, 0.753333, 0.733333, 0.736667, 0.743333, 0.743333, 0.740000, 0.723333, 0.743333],  # ADV
            [0.751270, 0.745000, 0.746280, 0.740750, 0.755740, 0.741060, 0.756580, 0.745570, 0.747090, 0.749340],  # CARE
            [0.766667, 0.766667, 0.766667, 0.763333, 0.766667, 0.766667, 0.766667, 0.763333, 0.766667, 0.766667],  # FMT_s
            [0.766667, 0.766667, 0.770000, 0.770000, 0.766667, 0.766667, 0.766667, 0.770000, 0.766667, 0.766667],  # FMT_d
        ]

    elif task == "German-Sex":
        data = [
            [0.740000, 0.743333, 0.746667, 0.770000, 0.753333, 0.733333, 0.710000, 0.716667, 0.720000, 0.726667],  # REW
            [0.753333, 0.670000, 0.756667, 0.763333, 0.726667, 0.763333, 0.756667, 0.750000, 0.686667, 0.770000],  # ADV
            [0.752356, 0.741500, 0.746723, 0.751233, 0.738371, 0.751455, 0.749623, 0.757263, 0.756543, 0.746497],  # CARE
            [0.773333, 0.770000, 0.770000, 0.770000, 0.770000, 0.773333, 0.770000, 0.773333, 0.770000, 0.770000],  # FMT_s
            [0.770000, 0.773333, 0.770000, 0.773333, 0.773333, 0.770000, 0.770000, 0.773333, 0.770000, 0.770000],  # FMT_d
        ]

    return data


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["adult", "bank", "compas", "german"], help="Dataset name"
    )
    parser.add_argument("-p", "--protected", type=str, required=True, help="Protected attribute")
    opt = parser.parse_args()

    return opt


def main(opt):
    print("\nplot accuracy")
    data = data_helper(opt.dataset, opt.protected)

    labels = ["REW", "ADV", "CARE", "FMT$_s$", "FMT$_d$"]
    colors = [
        (217 / 255.0, 33 / 255.0, 13 / 255.0),  # 正红
        (242 / 255.0, 149 / 255.0, 0 / 255.0),  # 橙色
        (255 / 255.0, 242 / 255.0, 0 / 255.0),  # 纯黄
        (0 / 255.0, 91 / 255.0, 171 / 255.0),  # 纯蓝
        (143 / 255.0, 7 / 255.0, 131 / 255.0),  # 紫色
    ]

    plt.figure(figsize=(9, 9))
    plt.rc("font", family="Times New Roman", size=32)

    bplot = plt.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    if opt.dataset == "adult":
        y_major_locator = plt.MultipleLocator(0.01)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0.81, 0.86)

    elif opt.dataset == "bank":
        y_major_locator = plt.MultipleLocator(0.01)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0.85, 0.90)

    elif opt.dataset == "compas":
        y_major_locator = plt.MultipleLocator(0.01)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0.63, 0.66)

    elif opt.dataset == "german":
        y_major_locator = plt.MultipleLocator(0.03)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0.66, 0.78)

    # plt.legend(bplot["boxes"], labels, loc="upper right")
    plt.savefig(fname="../RQ5_results/boxplot/boxplot_%s-%s_acc.pdf" % (opt.dataset, opt.protected), bbox_inches="tight")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
