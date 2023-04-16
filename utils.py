import json
import numpy as np

from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def combine_model(regression_model, source_model, input_shape, chosen_layer=3):
    new_model = Sequential()
    new_model.add(InputLayer(input_shape=input_shape))

    for layer in regression_model.layers:
        new_model.add(layer)

    for layer in source_model.layers[(chosen_layer + 1) :]:
        new_model.add(layer)

    return new_model


def get_groups(dataset, protected):
    if dataset == "adult":
        if protected == "sex":
            privileged_groups = [{"sex": 1}]
            unprivileged_groups = [{"sex": 0}]
        else:
            privileged_groups = [{"race": 1}]
            unprivileged_groups = [{"race": 0}]

    elif dataset == "bank":
        privileged_groups = [{"age": 1}]
        unprivileged_groups = [{"age": 0}]

    elif dataset == "compas":
        if protected == "sex":
            privileged_groups = [{"sex": 1}]
            unprivileged_groups = [{"sex": 0}]
        else:
            privileged_groups = [{"race": 1}]
            unprivileged_groups = [{"race": 0}]

    elif dataset == "german":
        if protected == "sex":
            privileged_groups = [{"sex": 1}]
            unprivileged_groups = [{"sex": 0}]
        else:
            privileged_groups = [{"age": 1}]
            unprivileged_groups = [{"age": 0}]

    return privileged_groups, unprivileged_groups


def measure_final_score(dataset_orig_test, dataset_orig_predict, privileged_groups, unprivileged_groups):

    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average="macro")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    f1score_macro = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    classified_metric_pred = ClassificationMetric(
        dataset_orig_test, dataset_orig_predict, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )

    spd = abs(classified_metric_pred.statistical_parity_difference())
    aaod = classified_metric_pred.average_abs_odds_difference()
    eod = abs(classified_metric_pred.equal_opportunity_difference())

    return accuracy, recall_macro, precision_macro, f1score_macro, mcc, spd, aaod, eod


def MLP(input_shape, nb_classes):
    model = Sequential(name="MLP")
    model.add(Dense(64, activation="relu", input_shape=input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(nb_classes, activation="softmax"))

    model.summary()

    return model


def reverse(x, sens_index):
    x_reverse = deepcopy(x)

    for i in range(len(x)):
        if x[i][sens_index] == 0:
            x_reverse[i][sens_index] = 1
        elif x[i][sens_index] == 1:
            x_reverse[i][sens_index] = 0

    return x_reverse


def sensitive_balance(x_mutation, gt, sens_distr, sens_index):
    index0 = []
    index1 = []
    for i, sample in enumerate(x_mutation):
        if x_mutation[i][sens_index] == 0:
            index0.append(i)
        else:
            index1.append(i)

    print(len(index0), len(index1))

    ratio_sens = float(sens_distr[0] / sens_distr[1])
    ratio_mutation = float(len(index0) / len(index1))
    print(ratio_sens, ratio_mutation)

    if ratio_sens > ratio_mutation:
        num = int((sens_distr[1] / sens_distr[0]) * len(index0))
        print(num)
        index1 = np.random.choice(index1, size=num, replace=False)
        index1 = index1.tolist()
    elif ratio_sens < ratio_mutation:
        num = int((sens_distr[0] / sens_distr[1]) * len(index1))
        print(num)
        index0 = np.random.choice(index0, size=num, replace=False)
        index0 = index0.tolist()

    index_new = index0 + index1
    index_new.sort()

    x_mutation = x_mutation[index_new]
    gt = gt[index_new]

    print(x_mutation.shape, gt.shape)

    return x_mutation, gt


def set_json(dataset, columns):
    feature2idx = {}
    idx2feature = {}

    for col in columns:
        if "=" in col:
            name = col.split("=")[0]
            value = col.split("=")[1]

            # feature to idx
            dict_f2i = feature2idx.setdefault(name, {})
            idx = len(dict_f2i)
            dict_f2i[value] = idx

            # idx to feature
            dict_i2f = idx2feature.setdefault(name, {})
            idx_str = str(len(dict_i2f))
            dict_i2f[idx_str] = value

    feature2idx_json = json.dumps(feature2idx, indent=4)
    with open("features/%s/feature2idx.json" % (dataset), "w") as json_file:
        json_file.write(feature2idx_json)

    idx2feature_json = json.dumps(idx2feature, indent=4)
    with open("features/%s/idx2feature.json" % (dataset), "w") as json_file:
        json_file.write(idx2feature_json)


def sub_model(source_model, chosen_layer=3):
    model = Sequential()
    pos = chosen_layer + 1

    for layer in source_model.layers[:pos]:
        layer.trainable = True
        model.add(layer=layer)

    return model
