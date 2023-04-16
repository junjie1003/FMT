import random
import numpy as np


def non_sensitive_distribution(x, non_sens, column_names):
    keys_list = []
    distr_list = []

    for idx in range(len(non_sens)):
        ns = column_names[non_sens[idx]]
        print(ns)

        dic = {}
        for i in range(len(x)):
            key = x[i][non_sens[idx]]
            dic[key] = dic.setdefault(key, 0) + 1

        keys = sorted(list(dic.keys()))
        print(keys)

        distr = [dic[k] for k in keys]

        keys_list.append(keys)
        distr_list.append(distr)

    return keys_list, distr_list


def sensitive_distribution(x, protected, sens_index):
    sens_dict = {"sex": 2, "race": 10, "age": 100}
    sens_distr = np.zeros(sens_dict[protected], dtype=int)
    for i in range(len(x)):
        index = int(x[i][sens_index])
        sens_distr[index] += 1

    return sens_distr


def random_index(rate):
    """
    A probability function of a random variable.
    - rate: list<int>
    - return: index of a probability event
    """

    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))

    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break

    return index


def mutation(dataset, protected, column_names, x_train, rand):
    sens_index = column_names.index(protected)
    print(dataset, protected, sens_index)

    non_sens = [i for i in range(x_train.shape[1])]
    non_sens.remove(sens_index)
    print(non_sens)

    keys_list, distr_list = non_sensitive_distribution(x_train, non_sens, column_names)

    sens_distr = sensitive_distribution(x_train, protected, sens_index)
    print(sens_distr)

    x_mutation = np.zeros((x_train.shape[0] * 2, x_train.shape[1]))

    print(rand)
    if rand == "distribution":
        for i in range(len(x_train)):
            for idx in range(len(non_sens)):
                keys = keys_list[idx]
                distr = distr_list[idx]
                pos = random_index(rate=distr)
                x_mutation[i][non_sens[idx]] = keys[pos]
                x_mutation[i + len(x_train)][non_sens[idx]] = keys[pos]
    elif rand == "random":
        for i in range(len(x_train)):
            for idx in range(len(non_sens)):
                keys = keys_list[idx]
                temp = str(np.random.choice(keys))
                x_mutation[i][non_sens[idx]] = temp
                x_mutation[i + len(x_train)][non_sens[idx]] = temp

    for i in range(len(x_train)):
        if protected == "age":
            x_mutation[i][sens_index] = 12
            x_mutation[i + len(x_train)][sens_index] = 60
        else:
            x_mutation[i][sens_index] = 0
            x_mutation[i + len(x_train)][sens_index] = 1

    return x_mutation
