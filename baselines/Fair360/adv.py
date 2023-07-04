import os
import sys
import argparse
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import tensorflow as tf

# import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
from itertools import product
from numpy import mean, std
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath("/root/FMT"))
from utils import get_groups, measure_final_score

EPOCHS = 500
nb_classes = 2


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["adult", "bank", "compas", "german"], help="Dataset name"
    )
    parser.add_argument("-p", "--protected", type=str, required=True, help="Protected attribute")
    opt = parser.parse_args()

    return opt


def func(learning_rate, batch_size):
    x_train = np.load("/root/FMT/data/processed/%s/%s_x_train.npy" % (opt.dataset, opt.dataset))
    y_train = np.load("/root/FMT/data/processed/%s/%s_y_train.npy" % (opt.dataset, opt.dataset))
    x_test = np.load("/root/FMT/data/processed/%s/%s_x_test.npy" % (opt.dataset, opt.dataset))
    y_test = np.load("/root/FMT/data/processed/%s/%s_y_test.npy" % (opt.dataset, opt.dataset))

    input_shape = (x_train.shape[1],)
    print(input_shape)

    column_names = open("/root/FMT/data/raw/%s/column_names" % opt.dataset, "r", encoding="UTF-8").read().splitlines()
    print(len(column_names))
    print(column_names)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    train_labels = []
    for i in range(len(y_train)):
        if y_train[i][0] == 1:
            train_labels.append(0)
        else:
            train_labels.append(1)
    train_labels = np.array(train_labels).reshape(-1, 1)
    train_data = np.c_[x_train, train_labels]

    test_labels = []
    for i in range(len(y_test)):
        if y_test[i][0] == 1:
            test_labels.append(0)
        else:
            test_labels.append(1)
    test_labels = np.array(test_labels).reshape(-1, 1)
    test_data = np.c_[x_test, test_labels]

    source_model = load_model("/root/FMT/models/%s_source_model.h5" % opt.dataset)
    y_pred = source_model.predict(x_test)

    predict_labels = []
    for i in range(len(y_pred)):
        if y_pred[i][0] >= 0.5:
            predict_labels.append(0)
        else:
            predict_labels.append(1)
    predict_labels = np.array(predict_labels).reshape(-1, 1)
    predict_data = np.c_[x_test, predict_labels]

    privileged_groups, unprivileged_groups = get_groups(opt.dataset, opt.protected)
    performance_index = ["accuracy", "recall", "precision", "f1score", "mcc", "spd", "aaod", "eod"]

    scaler = MinMaxScaler()
    scaler.fit(train_data)
    dataset_orig_train = pd.DataFrame(data=scaler.transform(train_data), columns=column_names)
    dataset_orig_test = pd.DataFrame(data=scaler.transform(test_data), columns=column_names)
    dataset_orig_predict = pd.DataFrame(data=scaler.transform(predict_data), columns=column_names)

    dataset_orig_train = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=dataset_orig_train,
        label_names=["probability"],
        protected_attribute_names=[opt.protected],
    )
    dataset_orig_test = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=dataset_orig_test,
        label_names=["probability"],
        protected_attribute_names=[opt.protected],
    )
    dataset_orig_predict = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=dataset_orig_predict,
        label_names=["probability"],
        protected_attribute_names=[opt.protected],
    )

    round_result_source = measure_final_score(dataset_orig_test, dataset_orig_predict, privileged_groups, unprivileged_groups)

    for i in range(len(performance_index)):
        print("%s-%s: %s=%f\n" % (opt.dataset, opt.protected, performance_index[i], round_result_source[i]))

    val_name = "RQ5_results/adv/adv_{}_{}_{}_{}.txt".format(opt.dataset, opt.protected, str(learning_rate), str(batch_size))
    fout = open(val_name, "w")

    results = {}
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 10
    for r in range(repeat_time):
        print(r)

        tf.reset_default_graph()
        sess = tf.Session()
        scope = "adv" + str(r)
        adversarial = AdversarialDebiasing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            scope_name=scope,
            sess=sess,
            batch_size=batch_size,
            debias=True,
        )
        adversarial = adversarial.fit(dataset_orig_train)
        pred_ad = adversarial.predict(dataset_orig_test)

        round_result = measure_final_score(dataset_orig_test, pred_ad, privileged_groups, unprivileged_groups)

        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    round_result_final = []

    for p_index in performance_index:
        fout.write(p_index + "\t")
        for i in range(repeat_time):
            fout.write("%f\t" % results[p_index][i])
        fout.write("%f\t%f\n" % (mean(results[p_index]), std(results[p_index])))
        round_result_final.append(mean(results[p_index]))

    fout.write("\n")

    for i in range(len(performance_index)):
        fout.write("%s change: %.6f\n" % (performance_index[i], round_result_final[i] - round_result_source[i]))

    fout.close()


def main(opt):
    if not os.path.exists("RQ5_results/adv"):
        os.makedirs("RQ5_results/adv")

    learning_rate = [1e-5, 1e-4, 1e-3]
    batch_size = [16, 32, 64, 128]

    params = list(product(learning_rate, batch_size))
    print(params)

    pool = multiprocessing.Pool(12)
    pool.starmap(func=func, iterable=params)
    pool.close()
    pool.join()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
