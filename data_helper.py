import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from mutation import mutation
from datasets import AdultDataset, BankDataset, CompasDataset, GermanDataset


def default_preprocessing_compas(df):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[
        (df.days_b_screening_arrest <= 30)
        & (df.days_b_screening_arrest >= -30)
        & (df.is_recid != -1)
        & (df.c_charge_degree != "O")
        & (df.score_text != "N/A")
    ]


def default_preprocessing_german(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {"A91": "male", "A93": "male", "A94": "male", "A92": "female", "A95": "female"}
    # df["sex"] = df["personal_status"].replace(status_map)
    df.insert(loc=0, column="sex", value=df["personal_status"].replace(status_map))

    return df


def preprocessing(
    df,
    label_name,
    protected_attribute_names,
    instance_weights_name,
    categorical_features,
    features_to_keep,
    features_to_drop,
    custom_preprocessing,
):

    # 1. Perform dataset-specific preprocessing
    if custom_preprocessing:
        df = custom_preprocessing(df)

    # 2. Drop unrequested columns
    features_to_keep = features_to_keep or df.columns.tolist()
    keep = set(features_to_keep) | set(protected_attribute_names) | set(categorical_features) | set([label_name])
    if instance_weights_name:
        keep |= set([instance_weights_name])
    df = df[sorted(keep - set(features_to_drop), key=df.columns.get_loc)]
    categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

    # 3. Remove any rows that have missing data
    df = df.dropna()

    return df


def get_data(dataset, protected, rand):
    if dataset == "adult":
        label_name = "income-per-year"
        protected_attribute_names = ["race", "sex"]
        instance_weights_name = None
        categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"]
        features_to_keep = []
        features_to_drop = ["fnlwgt"]
        na_values = ["?"]
        custom_preprocessing = None

        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "adult", "adult.data")
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "adult", "adult.test")

        # as given by adult.names
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income-per-year",
        ]

        train = pd.read_csv(train_path, header=None, names=column_names, skipinitialspace=True, na_values=na_values)
        test = pd.read_csv(test_path, header=0, names=column_names, skipinitialspace=True, na_values=na_values)

        df = pd.concat([test, train], ignore_index=True)
        print(df)

        df = preprocessing(
            df=df,
            label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            custom_preprocessing=custom_preprocessing,
        )
        print(df)

        column_names = df.columns.tolist()
        print(len(column_names))
        print(column_names)

        with open("features/adult/feature2idx.json", "r", encoding="UTF-8") as f:
            feature2idx = json.load(f)

        for feature in feature2idx:
            df[feature] = df[feature].map(feature2idx[feature])

        df_orig_train, df_orig_test = train_test_split(df, test_size=0.3, random_state=2022, shuffle=True)
        print(df_orig_train)
        print(df_orig_test)

        train_data = df_orig_train.to_numpy()
        test_data = df_orig_test.to_numpy()

        x_mutation = mutation(dataset, protected, column_names, train_data[:, :-1], rand)
        mutation_data = np.c_[x_mutation, np.zeros((len(x_mutation), 1))]

        df_train = pd.DataFrame(data=train_data, columns=column_names, dtype=int)
        df_test = pd.DataFrame(data=test_data, columns=column_names, dtype=int)
        df_mutation = pd.DataFrame(data=mutation_data, columns=column_names, dtype=int)
        print(df_train)
        print(df_test)
        print(df_mutation)

        with open("features/adult/idx2feature.json", "r", encoding="UTF-8") as f:
            idx2feature = json.load(f)

        idx2feature_new = {}
        for feature in idx2feature:
            temp = {}
            for key, value in idx2feature[feature].items():
                temp[int(key)] = value
            idx2feature_new[feature] = temp

        for feature in idx2feature_new:
            df_train[feature] = df_train[feature].map(idx2feature_new[feature])
            df_test[feature] = df_test[feature].map(idx2feature_new[feature])
            df_mutation[feature] = df_mutation[feature].map(idx2feature_new[feature])
        print(df_train)
        print(df_test)
        print(df_mutation)

        df_ensemble = pd.concat([df_train, df_test, df_mutation], ignore_index=True)

        dataset_orig = AdultDataset(df_data=df_ensemble).convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "probability")

    elif dataset == "bank":
        label_name = "y"
        protected_attribute_names = ["age"]
        instance_weights_name = None
        categorical_features = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ]
        features_to_keep = []
        features_to_drop = []
        na_values = ["unknown"]
        custom_preprocessing = None

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "bank", "bank-additional-full.csv")

        df = pd.read_csv(filepath, sep=";", na_values=na_values)
        print(df)

        df = preprocessing(
            df=df,
            label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            custom_preprocessing=custom_preprocessing,
        )
        print(df)

        column_names = df.columns.tolist()
        print(len(column_names))
        print(column_names)

        with open("features/bank/feature2idx.json", "r", encoding="UTF-8") as f:
            feature2idx = json.load(f)

        for feature in feature2idx:
            df[feature] = df[feature].map(feature2idx[feature])

        df_orig_train, df_orig_test = train_test_split(df, test_size=0.3, random_state=2022, shuffle=True)
        print(df_orig_train)
        print(df_orig_test)

        train_data = df_orig_train.to_numpy()
        test_data = df_orig_test.to_numpy()

        x_mutation = mutation(dataset, protected, column_names, train_data[:, :-1], rand)
        mutation_data = np.c_[x_mutation, np.zeros((len(x_mutation), 1))]

        df_train = pd.DataFrame(data=train_data, columns=column_names, dtype=int)
        df_test = pd.DataFrame(data=test_data, columns=column_names, dtype=int)
        df_mutation = pd.DataFrame(data=mutation_data, columns=column_names, dtype=int)
        print(df_train)
        print(df_test)
        print(df_mutation)

        with open("features/bank/idx2feature.json", "r", encoding="UTF-8") as f:
            idx2feature = json.load(f)

        idx2feature_new = {}
        for feature in idx2feature:
            temp = {}
            for key, value in idx2feature[feature].items():
                temp[int(key)] = value
            idx2feature_new[feature] = temp

        for feature in idx2feature_new:
            df_train[feature] = df_train[feature].map(idx2feature_new[feature])
            df_test[feature] = df_test[feature].map(idx2feature_new[feature])
            df_mutation[feature] = df_mutation[feature].map(idx2feature_new[feature])
        print(df_train)
        print(df_test)
        print(df_mutation)

        df_ensemble = pd.concat([df_train, df_test, df_mutation], ignore_index=True)

        dataset_orig = BankDataset(df_data=df_ensemble).convert_to_dataframe()[0]
        dataset_orig.rename(columns={"y": "probability"}, inplace=True)

    elif dataset == "compas":
        label_name = "two_year_recid"
        protected_attribute_names = ["sex", "race"]
        instance_weights_name = None
        categorical_features = ["age_cat", "c_charge_degree", "c_charge_desc"]
        features_to_keep = [
            "sex",
            "age",
            "age_cat",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "c_charge_degree",
            "c_charge_desc",
            "two_year_recid",
        ]
        features_to_drop = []
        na_values = []
        custom_preprocessing = default_preprocessing_compas

        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "raw", "compas", "compas-scores-two-years.csv"
        )

        df = pd.read_csv(filepath, index_col="id", na_values=na_values)
        print(df)

        df = preprocessing(
            df=df,
            label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            custom_preprocessing=custom_preprocessing,
        )
        print(df)

        column_names = df.columns.tolist()
        print(len(column_names))
        print(column_names)

        with open("features/compas/feature2idx.json", "r", encoding="UTF-8") as f:
            feature2idx = json.load(f)

        for feature in feature2idx:
            df[feature] = df[feature].map(feature2idx[feature])

        df_orig_train, df_orig_test = train_test_split(df, test_size=0.3, random_state=2022, shuffle=True)
        print(df_orig_train)
        print(df_orig_test)

        train_data = df_orig_train.to_numpy()
        test_data = df_orig_test.to_numpy()

        x_mutation = mutation(dataset, protected, column_names, train_data[:, :-1], rand)
        mutation_data = np.c_[x_mutation, np.zeros((len(x_mutation), 1))]

        df_train = pd.DataFrame(data=train_data, columns=column_names, dtype=int)
        df_test = pd.DataFrame(data=test_data, columns=column_names, dtype=int)
        df_mutation = pd.DataFrame(data=mutation_data, columns=column_names, dtype=int)
        print(df_train)
        print(df_test)
        print(df_mutation)

        with open("features/compas/idx2feature.json", "r", encoding="UTF-8") as f:
            idx2feature = json.load(f)

        idx2feature_new = {}
        for feature in idx2feature:
            temp = {}
            for key, value in idx2feature[feature].items():
                temp[int(key)] = value
            idx2feature_new[feature] = temp

        for feature in idx2feature_new:
            df_train[feature] = df_train[feature].map(idx2feature_new[feature])
            df_test[feature] = df_test[feature].map(idx2feature_new[feature])
            df_mutation[feature] = df_mutation[feature].map(idx2feature_new[feature])
        print(df_train)
        print(df_test)
        print(df_mutation)

        df_ensemble = pd.concat([df_train, df_test, df_mutation], ignore_index=True)

        dataset_orig = CompasDataset(df_data=df_ensemble).convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "probability")
        dataset_orig["probability"] = np.where(dataset_orig["probability"] == 1, 0, 1)

    elif dataset == "german":
        label_name = "credit"
        protected_attribute_names = ["sex", "age"]
        instance_weights_name = None
        categorical_features = [
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "other_debtors",
            "property",
            "installment_plans",
            "housing",
            "skill_level",
            "telephone",
            "foreign_worker",
        ]
        features_to_keep = []
        features_to_drop = ["personal_status"]
        na_values = []
        custom_preprocessing = default_preprocessing_german

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "german", "german.data")

        # as given by german.doc
        column_names = [
            "status",
            "month",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings",
            "employment",
            "investment_as_income_percentage",
            "personal_status",
            "other_debtors",
            "residence_since",
            "property",
            "age",
            "installment_plans",
            "housing",
            "number_of_credits",
            "skill_level",
            "people_liable_for",
            "telephone",
            "foreign_worker",
            "credit",
        ]

        df = pd.read_csv(filepath, sep=" ", header=None, names=column_names, na_values=na_values)
        print(df)

        df = preprocessing(
            df=df,
            label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            custom_preprocessing=custom_preprocessing,
        )
        print(df)

        column_names = df.columns.tolist()
        print(len(column_names))
        print(column_names)

        with open("features/german/feature2idx.json", "r", encoding="UTF-8") as f:
            feature2idx = json.load(f)

        for feature in feature2idx:
            df[feature] = df[feature].map(feature2idx[feature])

        df_orig_train, df_orig_test = train_test_split(df, test_size=0.3, random_state=2022, shuffle=True)
        print(df_orig_train)
        print(df_orig_test)

        train_data = df_orig_train.to_numpy()
        test_data = df_orig_test.to_numpy()

        x_mutation = mutation(dataset, protected, column_names, train_data[:, :-1], rand)
        mutation_data = np.c_[x_mutation, np.zeros((len(x_mutation), 1))]

        df_train = pd.DataFrame(data=train_data, columns=column_names, dtype=int)
        df_test = pd.DataFrame(data=test_data, columns=column_names, dtype=int)
        df_mutation = pd.DataFrame(data=mutation_data, columns=column_names, dtype=int)
        print(df_train)
        print(df_test)
        print(df_mutation)

        with open("features/german/idx2feature.json", "r", encoding="UTF-8") as f:
            idx2feature = json.load(f)

        idx2feature_new = {}
        for feature in idx2feature:
            temp = {}
            for key, value in idx2feature[feature].items():
                temp[int(key)] = value
            idx2feature_new[feature] = temp

        for feature in idx2feature_new:
            df_train[feature] = df_train[feature].map(idx2feature_new[feature])
            df_test[feature] = df_test[feature].map(idx2feature_new[feature])
            df_mutation[feature] = df_mutation[feature].map(idx2feature_new[feature])
        print(df_train)
        print(df_test)
        print(df_mutation)

        df_ensemble = pd.concat([df_train, df_test, df_mutation], ignore_index=True)

        dataset_orig = GermanDataset(df_data=df_ensemble).convert_to_dataframe()[0]
        dataset_orig["credit"] = np.where(dataset_orig["credit"] == 1, 1, 0)
        dataset_orig.columns = dataset_orig.columns.str.replace("credit", "probability")

    columns = dataset_orig.columns.to_list()
    print(len(columns))
    print(columns)

    with open("data/raw/%s/column_names" % dataset, "w", encoding="UTF-8") as f:
        for col in columns:
            f.write(col + "\n")

    dataset_orig_train = dataset_orig.iloc[0 : len(df_train)]
    dataset_orig_test = dataset_orig.iloc[len(df_train) : (len(df_train) + len(df_test))]
    dataset_orig_mutation = dataset_orig.iloc[(len(df_train) + len(df_test)) : (len(df_train) + len(df_test) + len(df_mutation))]
    dataset_orig_mutation.drop(dataset_orig_mutation.columns[-1], axis=1, inplace=True)
    print(dataset_orig_train)
    print(dataset_orig_test)
    print(dataset_orig_mutation)

    dataset_orig_train = dataset_orig_train.to_numpy()
    dataset_orig_test = dataset_orig_test.to_numpy()
    dataset_orig_mutation = dataset_orig_mutation.to_numpy()

    x_train = dataset_orig_train[:, :-1]
    print(x_train.shape)

    train_labels = dataset_orig_train[:, -1]
    y_train = []
    for label in train_labels:
        if label == 0:
            y_train.append([1, 0])
        else:
            y_train.append([0, 1])
    y_train = np.array(y_train, dtype=float)
    print(y_train.shape)

    if not os.path.exists("data/processed/%s/%s_x_train.npy" % (dataset, dataset)):
        np.save("data/processed/%s/%s_x_train.npy" % (dataset, dataset), x_train)
    if not os.path.exists("data/processed/%s/%s_y_train.npy" % (dataset, dataset)):
        np.save("data/processed/%s/%s_y_train.npy" % (dataset, dataset), y_train)

    x_test = dataset_orig_test[:, :-1]
    print(x_test.shape)

    test_labels = dataset_orig_test[:, -1]
    y_test = []
    for label in test_labels:
        if label == 0:
            y_test.append([1, 0])
        else:
            y_test.append([0, 1])
    y_test = np.array(y_test, dtype=float)
    print(y_test.shape)

    if not os.path.exists("data/processed/%s/%s_x_test.npy" % (dataset, dataset)):
        np.save("data/processed/%s/%s_x_test.npy" % (dataset, dataset), x_test)
    if not os.path.exists("data/processed/%s/%s_y_test.npy" % (dataset, dataset)):
        np.save("data/processed/%s/%s_y_test.npy" % (dataset, dataset), y_test)

    print(dataset_orig_mutation.shape)
    if rand == "random":
        if not os.path.exists("data/processed/%s/%s_x_mutation_rand_%s.npy" % (dataset, dataset, protected)):
            np.save("data/processed/%s/%s_x_mutation_rand_%s.npy" % (dataset, dataset, protected), dataset_orig_mutation)
    elif rand == "distribution":
        if not os.path.exists("data/processed/%s/%s_x_mutation_%s.npy" % (dataset, dataset, protected)):
            np.save("data/processed/%s/%s_x_mutation_%s.npy" % (dataset, dataset, protected), dataset_orig_mutation)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["adult", "bank", "compas", "german"], help="Dataset name"
    )
    parser.add_argument("-p", "--protected", type=str, required=True, help="Protected attribute")
    parser.add_argument(
        "-r",
        "--rand",
        type=str,
        required=True,
        choices=["distribution", "random"],
        help="The way to generate non-sensitive attributes",
    )
    opt = parser.parse_args()

    return opt


def main(opt):
    get_data(opt.dataset, opt.protected, opt.rand)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
