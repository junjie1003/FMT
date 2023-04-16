import time
import keras
import os
import keras.backend as K
from keras.models import load_model, Model
from keras.optimizers import RMSprop
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, Input, Reshape, MaxPooling2D

from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras import optimizers
from keras.models import Sequential

batch_size = 128
epochs = 100
iterations = 196

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.05
    return 0.01

def new_loss_fuction(y_actual,y_predicted):
    t=10
    y_t=K.softmax(y_predicted/t)
    y_a=K.softmax(y_actual)
    
    return K.categorical_crossentropy(y_a, y_t)
    
model2=load_model('alexnet_image_model.189.h5',custom_objects={'new_loss_fuction':new_loss_fuction})

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.load("cifar10_x_train.npy")
y_train = np.load("cifar10_y_train.npy")
x_val=np.load('cifar10_x_validation.npy')
y_val=np.load('cifar10_y_validation.npy')
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0
x_val = x_val.astype("float32")
x_val /= 255.0
y_test = keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
x_train_mean = np.mean(x_train, axis=0)
x_val -= x_train_mean
x_test -= x_train_mean
x_train -= x_train_mean

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)

#loss,acc=model2.evaluate(x_train,y_train)
#print(acc)

y=model2.predict(x_train)
print(y)

input_shape=(32,32,3)#3通道图像数据
num_class=10

def AlexNet8():
    inputs = Input(shape=(32, 32, 3))
    # inputs = Input(shape=(28,28,1))
    x = inputs
    x = Conv2D(filters=96, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu")(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu")(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = Flatten()(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(10, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=y)
    return model

model = AlexNet8()
model.summary()#显示模型结构

sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss=new_loss_fuction, optimizer=sgd, metrics=["accuracy"])
change_lr = LearningRateScheduler(scheduler)
save_dir = os.path.join(os.getcwd(), "for_heat_map")
model_name = "alexnet_dis_image_model_ce_re.{epoch:03d}.h5"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_acc", verbose=1, save_best_only=False)
cbks = [checkpoint, change_lr]
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
datagen.fit(x_train)
model.fit(datagen.flow(x_train, y, batch_size=128),epochs=100,
                    callbacks=cbks, validation_data=(x_test, y_test))
