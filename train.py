import argparse
import multiprocessing
import numpy as np
import os
import random
import tensorflow as tf
import warnings

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

from itertools import product
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from utils import MLP

seed = 2022
EPOCHS = 500
nb_classes = 2
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def training(learning_rate, batch_size):
    x_train = np.load("data/processed/%s/%s_x_train.npy" % (opt.dataset, opt.dataset))
    y_train = np.load("data/processed/%s/%s_y_train.npy" % (opt.dataset, opt.dataset))
    x_test = np.load("data/processed/%s/%s_x_test.npy" % (opt.dataset, opt.dataset))
    y_test = np.load("data/processed/%s/%s_y_test.npy" % (opt.dataset, opt.dataset))

    input_shape = (x_train.shape[1],)
    print(input_shape)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    callback = EarlyStopping(monitor="loss", patience=3, verbose=1, mode="min")
    model = MLP(input_shape=input_shape, nb_classes=nb_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.categorical_crossentropy
    metric = tf.keras.metrics.categorical_accuracy
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    print("Fit model on training data")
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=EPOCHS, verbose=2, callbacks=[callback])

    print("Evaluate model on test data")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(
        "\nDataset: %s, learning rate: %s, batch size: %d, test loss: %.6f, test accuracy: %.6f\n"
        % (opt.dataset, str(learning_rate), batch_size, loss, accuracy)
    )

    model.save("models/source/%s_source_model_%s_%d.h5" % (opt.dataset, str(learning_rate), batch_size))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["adult", "bank", "compas", "german"], help="Dataset name"
    )
    opt = parser.parse_args()

    return opt


def main(opt):
    if not os.path.exists("models/source"):
        os.makedirs("models/source")

    learning_rate = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    batch_size = [16, 32, 64, 128]

    params = list(product(learning_rate, batch_size))
    print(params)

    pool = multiprocessing.Pool(20)
    pool.starmap(func=training, iterable=params)
    pool.close()
    pool.join()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
