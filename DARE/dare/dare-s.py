from keras.models import Sequential
from keras.callbacks import TensorBoard
import numpy as np
import keras
from keras.models import load_model, Model
from keras.layers import Layer ,InputLayer
import keras.backend as K
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from keras.datasets import cifar10,cifar100
import random
import copy
from foolbox.criteria import TargetClassProbability,TargetClass
import foolbox
from tqdm import tqdm
import datetime
import sys
from keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

pos=0
times=0
lr_rate=0.000001

epoch=10
test_list=np.load('test_list.npy')
test_list=test_list.tolist()



choose_layer=11


def submodel(source_model):
    model = Sequential()

    pos=choose_layer+1
    if times!=0:
        print('ok')
        model.add(InputLayer(input_shape=(32,32,3)))
        pos-=1
    i =0

    for l in source_model.layers[:pos]:

        l.trainable = True

        model.add(l)
        i+=1

        
    return model

                   
def keras_custom_loss_fuction(y_actual,y_predicted):
 
    length=128
    sum=0
    for i in range(length):


        sum=sum+tf.reduce_mean(tf.square(dif))            
    custom_loss_value=sum/length
    return custom_loss_value
   
def combine_model(model,source_model):
    new_model = Sequential()
    new_model.add(InputLayer(input_shape=(32,32,3)))
    for l in model.layers:
        new_model.add(l)
    new_model.summary()
    for l in source_model.layers[(choose_layer+1):]:
        new_model.add(l)
    return new_model


def get_ats(model,x):
    all_n = model.predict(x.reshape(-1,32,32,3))
    return all_n

def sample(start, stop,length):
    start, stop = (int(start), int(stop))
    length = int(abs(length))
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list
   

def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels

def pre_processing(train_images, test_images,train2):
    images = np.concatenate((train2, test_images), axis = 0)
    mean, std = get_mean_std(images)

    for i in range(train_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
        train2[:, :, :, i] = (train2[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images,train2


def color_preprocessing(x_validation,x_train,x_test):
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_validation -=x_train_mean


    return x_validation,x_train, x_test
			
        
if __name__ == "__main__":

    source_model=load_model('/home/zhangyingyi/adv/alexnet_model.200.h5')

    source_model.summary()#
    
    x_validation=np.load('svhn_x_validation.npy')
    y_validation=np.load('svhn_y_validation.npy')
    x_train=np.load('svhn_x_train.npy')
    y_train=np.load('svhn_y_train.npy')
    x_test=np.load('svhn_x_test.npy')
    y_test=np.load('svhn_y_test.npy')
    y_test-=1
    y_train-=1
    y_validation-=1
    
    y_test = keras.utils.to_categorical(y_test, 10)
    y_validation = keras.utils.to_categorical(y_validation, 10)

    x_validation,x_train, x_test = color_preprocessing(x_validation,x_train, x_test)


    model=submodel(source_model)
    model.summary()

    rs=np.load('svhn_alexnet_13_rs.npy') 
    modify=np.load('svhn_alexnet_13_modify.npy') 

    ok=[]

    for i in range(36000):
        if modify[i]==1:
            ok.append(i)
    print(len(ok))
    ok=np.array(ok)
    c_in=x_validation[ok]
    c_out=rs
    adam = optimizers.Adam(lr=lr_rate)
    model.compile(loss=keras.losses.MeanSquaredError(),optimizer=adam,metrics = ['categorical_accuracy'])

    save_dir = os.path.join(os.getcwd(), 'svhn_alexnet_delta-slice_finetune_'+str(lr_rate))
    model_name = 'model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=2,
                             save_best_only=False)
    cbks = [checkpoint]
    model.fit(x=c_in, y=c_out,epochs=epoch,
                    callbacks=cbks, verbose=2)
