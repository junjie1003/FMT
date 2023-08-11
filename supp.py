import os
import random
import argparse
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

from numpy import mean, std
from itertools import product
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses, metrics, optimizers

from utils import get_groups, measure_final_score, MLP

seed = 2022
EPOCHS = 500
nb_classes = 2
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def training(learning_rate, batch_size):
    x_train = np.load("data/processed/%s/%s_x_train_supp.npy" % (opt.dataset, opt.dataset))
    y_train = np.load("data/processed/%s/%s_y_train.npy" % (opt.dataset, opt.dataset))
    x_test = np.load("data/processed/%s/%s_x_test_supp.npy" % (opt.dataset, opt.dataset))
    y_test = np.load("data/processed/%s/%s_y_test.npy" % (opt.dataset, opt.dataset))

    input_shape = (x_train.shape[1],)
    print(input_shape)

    column_names = open("data/raw/%s/column_names" % opt.dataset, "r", encoding="UTF-8").read().splitlines()
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

    source_model = load_model("models/%s_source_model.h5" % opt.dataset)
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

    if not os.path.exists("./RQ5_results/supp"):
        os.makedirs("./RQ5_results/supp")

    val_name = "./RQ5_results/supp/supp_{}_{}_{}_{}.txt".format(
        opt.dataset, opt.protected, str(learning_rate), str(batch_size)
    )
    fout = open(val_name, "w")

    results = {}
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 10
    for r in range(repeat_time):
        print(r)

        callback = EarlyStopping(monitor="loss", patience=3, verbose=1, mode="min")
        model = MLP(input_shape=input_shape, nb_classes=nb_classes)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = losses.categorical_crossentropy
        metric = metrics.categorical_accuracy
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        print("Fit model on training data")
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=EPOCHS, verbose=2, callbacks=[callback])

        y_pred = model.predict(dataset_orig_test.features)

        predict_labels = []
        for i in range(len(y_pred)):
            if y_pred[i][0] >= 0.5:
                predict_labels.append(0)
            else:
                predict_labels.append(1)
        predict_labels = np.array(predict_labels).reshape(-1, 1)
        dataset_orig_predict.labels = predict_labels

        round_result = measure_final_score(dataset_orig_test, dataset_orig_predict, privileged_groups, unprivileged_groups)

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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["adult", "bank", "compas", "german"], help="Dataset name"
    )
    parser.add_argument("-p", "--protected", type=str, required=True, help="Protected attribute")
    opt = parser.parse_args()

    return opt


def main(opt):
    if not os.path.exists("models/supp"):
        os.makedirs("models/supp")

    learning_rate = [1e-5, 1e-4, 1e-3]
    batch_size = [16, 32, 64, 128]

    params = list(product(learning_rate, batch_size))
    print(params)

    pool = multiprocessing.Pool(12)
    pool.starmap(func=training, iterable=params)
    pool.close()
    pool.join()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)