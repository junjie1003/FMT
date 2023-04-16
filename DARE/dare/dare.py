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
    arr=np.load('current_key_svhn_vgg19.npy')
    length=len(arr)
    print(arr.shape)
    
    if length==0:
        length=1
        arr=np.ones((1,input_a,input_b,input_c), dtype=float)
    sum=0
    for i in range(length):
        a=(arr[i]==1).reshape(input_a,input_b,input_c)
        dif=y_actual[i]-y_predicted[i]
        d=dif[a]
        sum=sum+tf.reduce_mean(tf.square(d))            
    custom_loss_value=sum/length
    return custom_loss_value

def get_weight(new_w):
    
    pos=np.load('svhn_alexnet_slice_pos.npy')
    val=np.load('svhn_alexnet_slice_val.npy')
    rs = deepcopy(val)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(96):
                    if pos[0][i][j][k][l] == 1:
                        rs[0][i][j][k][l] = new_w[0][i][j][k][l]
    
    for i in range(3):
        for j in range(3):
            for k in range(96):
                for l in range(256):
                    if pos[1][i][j][k][l] == 1:
                        rs[1][i][j][k][l] = new_w[1][i][j][k][l]
    for i in range(3):
        for j in range(3):
            for k in range(256):
                for l in range(384):
                    if pos[2][i][j][k][l] == 1:
                        rs[2][i][j][k][l] = new_w[2][i][j][k][l]
    for i in range(3):
        for j in range(3):
            for k in range(384):
                for l in range(384):
                    if pos[3][i][j][k][l] == 1:
                        rs[3][i][j][k][l] = new_w[3][i][j][k][l]
    for i in range(3):
        for j in range(3):
            for k in range(384):
                for l in range(256):
                    if pos[4][i][j][k][l] == 1:
                        rs[4][i][j][k][l] = new_w[4][i][j][k][l]
    return rs

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

    source_model.summary()
    
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
    for e in range(epoch):
        
        print('epoch:',e)
        start = datetime.datetime.now()

        if times==0:
            m=source_model
        else:
            m=new_model
        m.compile(loss=keras.losses.MeanSquaredError(),optimizer='adam',metrics = ['categorical_accuracy'])

        random_list1=sample(0,len(ok)-1,128)

        ok=np.array(ok)
        c_list=ok[random_list1]
        c_in=X_train[c_list]
        c_error_y=Y_train[c_list]
        c_error_y=c_error_y.reshape(-1)

        choose_truth=c_error_y
        
        times+=1
        c_out=rs[c_list]

        model.compile(loss=keras_custom_loss_fuction,optimizer='adam',metrics = ['accuracy'])
        K.set_value(model.optimizer.lr,  0.00001)
        c_error_y = keras.utils.to_categorical(c_error_y, 10)
        loss6,acc6=m.evaluate(c_in,c_error_y)
        loss=model.train_on_batch(x=c_in,y=c_out)
        print('loss:',loss)
        
        new_w = []
        new_b = []
        i=0
        for layer in model.layers:
            weights = layer.get_weights()
            if i==1 or i==5 or i==9 or i==10 or i==11:
                w = weights[0]
                b = weights[1]
                new_w.append(w)
                new_b.append(b)
        change_w = get_weight(new_w)
        i = 0
        j=0
        for layer in model.layers:
            wei = []
            if i==1 or i==5 or i==9 or i==10 or i==11:
                cw = np.array(change_w[j])
                wei.append(cw)
                b = np.array(new_b[j])
                wei.append(b)

                layer.set_weights(wei)
                j+=1
            i += 1
            
        new_model=combine_model(model,source_model)

        model=submodel(new_model)
        
    new_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['categorical_accuracy'])
    

    
    new_model.save_weights('svhn_alexnet_fintune.h5')
