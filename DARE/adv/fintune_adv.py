import tensorflow as tf
import keras
import os
import numpy as np
from matplotlib import pyplot as plt
from keras import Model
from keras.datasets import cifar10,cifar100
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras import optimizers
import keras.backend as K
import math
import random
from keras.models import load_model, Model

batch_size = 128

test_list=np.load('test_list.npy')
test_list=test_list.tolist()
validation_list=np.load('validation_list.npy')
validation_list=validation_list.tolist()

np.save('validation_list.npy',validation_list)
import argparse
parser = argparse.ArgumentParser(description="Experiments Script For DARE")
parser.add_argument("--adv", type=str, default='bim')
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dataset", type=str, default='svhn')
parser.add_argument("--model", type=str, default='alexnet')
params = parser.parse_args()
    
adv_name = params.adv
lr_rate = params.lr 
data_name = params.dataset
model_name = params.model

import time
start_clock = time.clock()
start_time = time.time()

def scheduler(epoch):
    #if epoch < 80:
    #    return lr_rate
    #if epoch < 160:
    #    return lr_rate
    return lr_rate
    
def color_preprocessing(x_validation,x_train,x_test):
    
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_validation -= x_train_mean 

    return x_validation,x_train, x_test

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

if data_name == 'cifar10':
    class_num = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train=np.load('cifar10_x_train.npy')
    y_train=np.load('cifar10_y_train.npy')
    x_validation=np.load('cifar10_x_validation.npy')
    y_validation=np.load('cifar10_y_validation.npy')
    y_test = keras.utils.to_categorical(y_test, class_num)
    y_train = keras.utils.to_categorical(y_train, class_num)
    y_validation = keras.utils.to_categorical(y_validation, class_num)
    x_train = x_train.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

elif data_name == 'fashion_mnist':
    class_num = 10
    x_test=np.load('fashion_mnist_x_test.npy')
    y_test=np.load('fashion_mnist_y_test.npy')
    x_train=np.load('fashion_mnist_x_train.npy')
    y_train=np.load('fashion_mnist_y_train.npy')
    x_validation=np.load('fashion_mnist_x_validation.npy')
    y_validation=np.load('fashion_mnist_y_validation.npy')
    y_test = keras.utils.to_categorical(y_test, class_num)
    y_validation = keras.utils.to_categorical(y_validation, class_num)
    x_train = x_train.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

elif data_name == 'svhn':
    class_num = 10
    x_test=np.load('svhn_x_test.npy')
    y_test=np.load('svhn_y_test.npy')
    x_train=np.load('svhn_x_train.npy')
    y_train=np.load('svhn_y_train.npy')
    x_validation=np.load('svhn_x_validation.npy')
    y_validation=np.load('svhn_y_validation.npy')
    y_test-=1
    y_train-=1
    y_validation-=1
    y_test = keras.utils.to_categorical(y_test, class_num)
    y_validation = keras.utils.to_categorical(y_validation, class_num)
    x_train = x_train.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    

if model_name == 'alexnet':
    x_validation, x_train, x_test = color_preprocessing(x_validation,x_train, x_test)

else:
    x_validation, x_test,x_train=pre_processing(x_validation, x_test,x_train)

if data_name == 'cifar10' and model_name == 'vgg16':
    model = load_model('model/cifar10_vgg16.200.hdf5')
elif data_name == 'svhn' and model_name == 'vgg16':
    model = load_model('svhn/vgg16_model.200.hdf5')
elif data_name == 'fashion_mnist' and model_name == 'vgg16':
    model = load_model('fashion_mnist/vgg16_model.200.hdf5')
elif data_name == 'cifar10' and model_name == 'vgg19':
    model = load_model('cifar10/model_vgg19.200.hdf5')
elif data_name == 'svhn' and model_name == 'vgg19':
    model = load_model('svhn/model_vgg19.200.hdf5')
elif data_name == 'fashion_mnist' and model_name == 'vgg19':
    model = load_model('fashion_mnist/model_vgg19.200.hdf5')
elif data_name == 'cifar10' and model_name == 'alexnet':
    model = load_model('cifar10/alexnet_model.200.h5')
elif data_name == 'svhn' and model_name == 'alexnet':
    model = load_model('svhn/alexnet_model.200.h5')
elif data_name == 'fashion_mnist' and model_name == 'alexnet':
    model = load_model('fashion_mnist/alexnet_model.200.h5')

label_address = adv_name+'_'+data_name+'_'+model_name+'_label.npy'
image_address = adv_name+'_'+data_name+'_'+model_name+'_image.npy'

bim_x=np.load(image_address)
bim_y=np.load(label_address)
bim_y = keras.utils.to_categorical(bim_y, class_num)

bim_train_x=[]
bim_train_y=[]
bim_validation_x=[]
bim_validation_y=[]
bim_test_x=[]
bim_test_y=[]
for i in range(len(bim_y)):
    if math.isnan(bim_x[i][0][0][0]):
        continue
    if i in test_list:
        bim_test_x.append(bim_x[i])
        bim_test_y.append(bim_y[i])
    elif i in validation_list:
        bim_validation_x.append(bim_x[i])
        bim_validation_y.append(bim_y[i])
    else:
        bim_train_x.append(bim_x[i])
        bim_train_y.append(bim_y[i])
bim_test_x=np.array(bim_test_x)
bim_test_y=np.array(bim_test_y) 
bim_validation_x=np.array(bim_validation_x)
bim_validation_y=np.array(bim_validation_y) 
bim_train_x=np.array(bim_train_x)
bim_train_y=np.array(bim_train_y)       
sgd = optimizers.SGD(lr=lr_rate, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=lr_rate)

if model_name == 'vgg16' or 'vgg19':
    frozen_layer = 20
else:
    frozen_layer = 5
p=0  
for layer in model.layers:
    if p<frozen_layer:# value for vgg
        layer.trainable = False
    else:
        layer.trainable = True
    p+=1
print(p)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


save_dir = os.path.join(os.getcwd(), adv_name+'_finetune_'+str(lr_rate))
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
change_lr = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=2,
                             save_best_only=False)
cbks = [checkpoint,change_lr]
print('Using real-time data augmentation.')
print(np.max(x_validation))
print(np.max(x_test))
print(np.max(x_train))

print(np.max(bim_test_x))

model.fit(x=bim_train_x, y=bim_train_y,epochs=20,
                    callbacks=cbks, verbose=2, validation_data=(bim_test_x, bim_test_y))

end_clock = time.clock()
end_time = time.time()
print('time')
print(end_clock-start_clock)
print(end_time-start_time)
