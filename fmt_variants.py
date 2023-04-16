import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

from copy import deepcopy
from numpy import mean, std
from keras import backend as K
from keras.models import load_model
from keras import losses, metrics, optimizers
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import MinMaxScaler

from utils import combine_model, get_groups, measure_final_score, reverse, sub_model


EPOCHS = 50


def slice_model(source_model, x_train, x_reverse, input_shape, chosen_layer=3):
    important_pos = []
    inp = source_model.input

    count = []
    ind1 = np.zeros(64, dtype=int)
    ind2 = np.zeros(32, dtype=int)
    ind3 = np.zeros(16, dtype=int)
    ind4 = np.zeros(8, dtype=int)
    count.append(ind1)
    count.append(ind2)
    count.append(ind3)
    count.append(ind4)

    outputs = [layer.output for layer in source_model.layers]
    functors = [K.function([inp], [out]) for out in outputs]

    for i in range(len(x_train)):
        test1 = x_train[i].reshape(1, input_shape[0])
        test2 = x_reverse[i].reshape(1, input_shape[0])

        layer_outs1 = [func([test1]) for func in functors]
        layer_outs2 = [func([test2]) for func in functors]

        for j in range(chosen_layer + 1):
            out1 = np.array(layer_outs1[j])
            out2 = np.array(layer_outs2[j])
            delta = abs(out1 - out2)
            shape = len(count[j])
            delta = delta.reshape(
                shape,
            )
            num = int(len(count[j]) * 2 / 10)
            z = delta.argsort()[-num:][::-1]

            for k in range(num):
                count[j][z[k]] += 1

    print(count)

    for i in range(chosen_layer + 1):
        num = int(round(len(count[i]) / 10))
        z = count[i].argsort()[-num:][::-1]
        important_pos.append(z)

    print(important_pos)

    return important_pos


def change_weights(important_pos, input_shape):
    cw = []
    ind1 = np.zeros((input_shape[0], 64), dtype=int)
    ind2 = np.zeros((64, 32), dtype=int)
    ind3 = np.zeros((32, 16), dtype=int)
    ind4 = np.zeros((16, 8), dtype=int)
    cw.append(ind1)
    cw.append(ind2)
    cw.append(ind3)
    cw.append(ind4)

    # 1
    for i in range(input_shape[0]):
        for j in important_pos[0]:
            cw[0][i][j] = 1
    # 2
    for i in important_pos[0]:
        for j in important_pos[1]:
            cw[1][i][j] = 1
    # 3
    for i in important_pos[1]:
        for j in important_pos[2]:
            cw[2][i][j] = 1
    # 4
    for i in important_pos[2]:
        for j in important_pos[3]:
            cw[3][i][j] = 1

    return cw


def get_weight(w_all, new_w, change, input_shape):
    rs = deepcopy(w_all)

    for i in range(input_shape[0]):
        for j in range(64):
            if change[0][i][j] == 1:
                rs[0][i][j] = new_w[0][i][j]

    for i in range(64):
        for j in range(32):
            if change[1][i][j] == 1:
                rs[1][i][j] = new_w[1][i][j]

    for i in range(32):
        for j in range(16):
            if change[2][i][j] == 1:
                rs[2][i][j] = new_w[2][i][j]

    for i in range(16):
        for j in range(8):
            if change[3][i][j] == 1:
                rs[3][i][j] = new_w[3][i][j]

    return rs


def conf_single(source_model, regression_model, x_train, x_mutation):
    pred = source_model.predict(x_mutation)
    print("pred shape: " + str(pred.shape))

    gt = regression_model.predict(x_mutation)
    print("gt shape: " + str(gt.shape))

    indices = []

    for i in range(len(x_train)):
        conf0 = np.max(pred[i])
        conf1 = np.max(pred[i + len(x_train)])
        if conf0 > conf1:
            gt[i + len(x_train)] = gt[i]
            indices.append(i + len(x_train))
        else:
            gt[i] = gt[i + len(x_train)]
            indices.append(i)

    x_mutation = x_mutation[indices]
    gt = gt[indices]

    print(x_mutation.shape, gt.shape)

    return x_mutation, gt


def conf_dual(source_model, regression_model, x_train, x_mutation):
    pred = source_model.predict(x_mutation)
    print("pred shape: " + str(pred.shape))

    gt = regression_model.predict(x_mutation)
    print("gt shape: " + str(gt.shape))

    for i in range(len(x_train)):
        conf0 = np.max(pred[i])
        conf1 = np.max(pred[i + len(x_train)])
        if conf0 > conf1:
            gt[i + len(x_train)] = gt[i]
        else:
            gt[i] = gt[i + len(x_train)]

    print(x_mutation.shape, gt.shape)

    return x_mutation, gt


def FMT_remove_s(dataset, protected, method, learning_rate, batch_size, input_shape, repeat_index):
    """
    Fairness improvement by Model Transformation (remove model slicing process).
    """
    source_model = load_model("models/%s_source_model.h5" % dataset)

    column_names = open("data/raw/%s/column_names" % dataset, "r", encoding="UTF-8").read().splitlines()
    print(len(column_names))
    print(column_names)

    x_train = np.load("data/processed/%s/%s_x_train.npy" % (dataset, dataset))
    y_train = np.load("data/processed/%s/%s_y_train.npy" % (dataset, dataset))
    x_test = np.load("data/processed/%s/%s_x_test.npy" % (dataset, dataset))
    y_test = np.load("data/processed/%s/%s_y_test.npy" % (dataset, dataset))
    x_mutation = np.load("data/processed/%s/%s_x_mutation_%s.npy" % (dataset, dataset, protected))

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_mutation = scaler.transform(x_mutation)

    regression_model = sub_model(source_model)
    optimizer = optimizers.Adam(learning_rate)
    loss = losses.mean_squared_error
    metric = metrics.categorical_accuracy
    regression_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    if method == "single":
        x_mutation, gt = conf_single(source_model, regression_model, x_train, x_mutation)
    elif method == "dual":
        x_mutation, gt = conf_dual(source_model, regression_model, x_train, x_mutation)

    for epoch in range(EPOCHS):
        print("Epoch %d\n" % epoch)

        regression_model.fit(x=x_mutation, y=gt, batch_size=batch_size, epochs=1, verbose=2)

    new_model = combine_model(regression_model, source_model, input_shape)
    loss = losses.categorical_crossentropy
    metric = metrics.categorical_accuracy
    new_model.compile(optimizer="adam", loss=loss, metrics=[metric])

    loss, accuracy = new_model.evaluate(x_test, y_test)
    print(loss)
    print(accuracy)

    save_dir = os.path.join(os.getcwd(), "models", "fmt_remove_s")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    new_model.save(
        "%s/fmt_remove_s_%s_%s_%s_%s_%s_%s.h5"
        % (save_dir, method, dataset, protected, str(learning_rate), str(batch_size), repeat_index)
    )


def FMT_remove_sl(dataset, protected, method, learning_rate, batch_size, repeat_index):
    """
    Fairness improvement by Model Transformation (remove model slicing and model transformation).
    """
    source_model = load_model("models/%s_source_model.h5" % dataset)

    column_names = open("data/raw/%s/column_names" % dataset, "r", encoding="UTF-8").read().splitlines()
    print(len(column_names))
    print(column_names)

    x_train = np.load("data/processed/%s/%s_x_train.npy" % (dataset, dataset))
    y_train = np.load("data/processed/%s/%s_y_train.npy" % (dataset, dataset))
    x_test = np.load("data/processed/%s/%s_x_test.npy" % (dataset, dataset))
    y_test = np.load("data/processed/%s/%s_y_test.npy" % (dataset, dataset))
    x_mutation = np.load("data/processed/%s/%s_x_mutation_%s.npy" % (dataset, dataset, protected))

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_mutation = scaler.transform(x_mutation)

    if method == "single":
        x_mutation, gt = conf_single(source_model, source_model, x_train, x_mutation)
    elif method == "dual":
        x_mutation, gt = conf_dual(source_model, source_model, x_train, x_mutation)

    new_model = load_model("models/%s_source_model.h5" % dataset)
    optimizer = optimizers.Adam(learning_rate)
    loss = losses.categorical_crossentropy
    metric = metrics.categorical_accuracy
    new_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    for epoch in range(EPOCHS):
        print("Epoch %d\n" % epoch)

        new_model.fit(x=x_mutation, y=gt, batch_size=batch_size, epochs=1, verbose=2)

    loss, accuracy = new_model.evaluate(x_test, y_test)
    print(loss)
    print(accuracy)

    save_dir = os.path.join(os.getcwd(), "models", "fmt_remove_sl")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    new_model.save(
        "%s/fmt_remove_sl_%s_%s_%s_%s_%s_%s.h5"
        % (save_dir, method, dataset, protected, str(learning_rate), str(batch_size), repeat_index)
    )


def FMT_rand(dataset, protected, method, learning_rate, batch_size, input_shape, repeat_index):
    """
    Fairness improvement by Model Transformation (non-sensitive attributes are randomly generated).
    """
    source_model = load_model("models/%s_source_model.h5" % dataset)

    column_names = open("data/raw/%s/column_names" % dataset, "r", encoding="UTF-8").read().splitlines()
    print(len(column_names))
    print(column_names)

    sens_index = column_names.index(protected)

    x_train = np.load("data/processed/%s/%s_x_train.npy" % (dataset, dataset))
    y_train = np.load("data/processed/%s/%s_y_train.npy" % (dataset, dataset))
    x_test = np.load("data/processed/%s/%s_x_test.npy" % (dataset, dataset))
    y_test = np.load("data/processed/%s/%s_y_test.npy" % (dataset, dataset))
    x_mutation = np.load("data/processed/%s/%s_x_mutation_rand_%s.npy" % (dataset, dataset, protected))

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_mutation = scaler.transform(x_mutation)

    x_reverse = reverse(x_train, sens_index)
    important_pos = slice_model(source_model, x_train, x_reverse, input_shape)

    regression_model = sub_model(source_model)
    optimizer = optimizers.Adam(learning_rate)
    loss = losses.mean_squared_error
    metric = metrics.categorical_accuracy
    regression_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    if method == "single":
        x_mutation, gt = conf_single(source_model, regression_model, x_train, x_mutation)
    elif method == "dual":
        x_mutation, gt = conf_dual(source_model, regression_model, x_train, x_mutation)

    w_all = []
    b_all = []
    for layer in regression_model.layers:
        weights = layer.get_weights()
        w = weights[0].tolist()
        b = weights[1].tolist()
        w_all.append(w)
        b_all.append(b)

    change = change_weights(important_pos, input_shape)
    for epoch in range(EPOCHS):
        print("Epoch %d\n" % epoch)

        regression_model.fit(x=x_mutation, y=gt, batch_size=batch_size, epochs=1, verbose=2)
        new_w = []
        new_b = []
        for layer in regression_model.layers:
            weights = layer.get_weights()
            w = weights[0].tolist()
            b = weights[1].tolist()
            new_w.append(w)
            new_b.append(b)
        change_w = get_weight(w_all, new_w, change, input_shape)

        i = 0
        for layer in regression_model.layers:
            wei = []
            cw = np.array(change_w[i])
            wei.append(cw)

            b = np.array(new_b[i])
            wei.append(b)

            layer.set_weights(wei)
            i += 1

    new_model = combine_model(regression_model, source_model, input_shape)
    loss = losses.categorical_crossentropy
    metric = metrics.categorical_accuracy
    new_model.compile(optimizer="adam", loss=loss, metrics=[metric])

    loss, accuracy = new_model.evaluate(x_test, y_test)
    print(loss)
    print(accuracy)

    save_dir = os.path.join(os.getcwd(), "models", "fmt_rand")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    new_model.save(
        "%s/fmt_rand_%s_%s_%s_%s_%s_%s.h5"
        % (save_dir, method, dataset, protected, str(learning_rate), str(batch_size), repeat_index)
    )


def get_fairness(learning_rate, batch_size):
    x_train = np.load("data/processed/%s/%s_x_train.npy" % (opt.dataset, opt.dataset))
    y_train = np.load("data/processed/%s/%s_y_train.npy" % (opt.dataset, opt.dataset))
    x_test = np.load("data/processed/%s/%s_x_test.npy" % (opt.dataset, opt.dataset))
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

    val_name = "RQ6&7_results/{}/fmt_{}_{}_{}_{}_{}_{}.txt".format(
        opt.variant, opt.variant, opt.method, opt.dataset, opt.protected, str(learning_rate), str(batch_size)
    )
    fout = open(val_name, "w")

    results = {}
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 10
    for r in range(repeat_time):
        print(r)

        if opt.variant == "remove_s":
            FMT_remove_s(opt.dataset, opt.protected, opt.method, learning_rate, batch_size, input_shape, repeat_index=r)
        elif opt.variant == "remove_sl":
            FMT_remove_sl(opt.dataset, opt.protected, opt.method, learning_rate, batch_size, repeat_index=r)
        elif opt.variant == "rand":
            FMT_rand(opt.dataset, opt.protected, opt.method, learning_rate, batch_size, input_shape, repeat_index=r)

        new_model = load_model(
            "models/fmt_%s/fmt_%s_%s_%s_%s_%s_%s_%s.h5"
            % (opt.variant, opt.variant, opt.method, opt.dataset, opt.protected, str(learning_rate), str(batch_size), r)
        )

        y_pred = new_model.predict(dataset_orig_test.features)

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
    parser.add_argument("-m", "--method", type=str, required=True, choices=["single", "dual"], help="Mutation method")
    parser.add_argument(
        "-v", "--variant", type=str, required=True, choices=["remove_s", "remove_sl", "rand"], help="Variants of FMT"
    )
    opt = parser.parse_args()

    return opt


def main(opt):
    if not os.path.exists("RQ6&7_results/remove_s"):
        os.makedirs("RQ6&7_results/remove_s")

    if not os.path.exists("RQ6&7_results/remove_sl"):
        os.makedirs("RQ6&7_results/remove_sl")

    if not os.path.exists("RQ6&7_results/rand"):
        os.makedirs("RQ6&7_results/rand")

    task = opt.dataset.capitalize() + "-" + opt.protected.capitalize()
    method = opt.method

    if task == "Adult-Race":
        if method == "single":
            learning_rate = 1e-05
            batch_size = 32
        elif method == "dual":
            learning_rate = 0.001
            batch_size = 32

    elif task == "Adult-Sex":
        if method == "single":
            learning_rate = 1e-05
            batch_size = 128
        elif method == "dual":
            learning_rate = 0.001
            batch_size = 16

    elif task == "Bank-Age":
        if method == "single":
            learning_rate = 1e-05
            batch_size = 32
        elif method == "dual":
            learning_rate = 1e-05
            batch_size = 128

    elif task == "Compas-Race":
        if method == "single":
            learning_rate = 0.0001
            batch_size = 128
        elif method == "dual":
            learning_rate = 0.001
            batch_size = 16

    elif task == "Compas-Sex":
        if method == "single":
            learning_rate = 1e-05
            batch_size = 16
        elif method == "dual":
            learning_rate = 1e-05
            batch_size = 64

    elif task == "German-Age":
        if method == "single":
            learning_rate = 0.001
            batch_size = 128
        elif method == "dual":
            learning_rate = 0.001
            batch_size = 128

    elif task == "German-Sex":
        if method == "single":
            learning_rate = 0.001
            batch_size = 64
        elif method == "dual":
            learning_rate = 0.0001
            batch_size = 16

    get_fairness(learning_rate, batch_size)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
