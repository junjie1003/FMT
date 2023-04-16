from keras import backend as K
import numpy as np
from keras.models import load_model, Model
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.regularizers import l2
#from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import SGD, Adam
import keras
from keras.models import Sequential
import tensorflow as tf
from keras.datasets import cifar10,cifar100
import time
import datetime

dataset='cifar10'
model_name='alexnet'
choose_layer=18

def submodel(source_model):
    model = Sequential()
    pos=choose_layer
    pos-=1
    i =0
    for l in source_model.layers[:pos]:

        l.trainable = True
        model.add(l)
        i+=1
        
    return model


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
    
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_validation -=x_train_mean


    return x_validation,x_train, x_test


model=load_model('model/cifar10_alexnet.200.h5')
if dataset=='cifar10':
    x_train=np.load('cifar10_x_train.npy')
    y_train=np.load('cifar10_y_train.npy')
    x_validation=np.load('cifar10_x_validation.npy')
    y_validation=np.load('cifar10_y_validation.npy')
    x_test=np.load('cifar10_x_test.npy')
    y_test=np.load('cifar10_y_test.npy')
elif dataset=='svhn':
    x_train=np.load('svhn_x_train.npy')
    y_train=np.load('svhn_y_train.npy')
    x_validation=np.load('svhn_x_validation.npy')
    y_validation=np.load('svhn_y_validation.npy')
    x_test=np.load('svhn_x_test.npy')
    y_test=np.load('svhn_y_test.npy')
    y_test-=1
    y_validation-=1
    y_train-=1
elif dataset=='fm':
    x_train=np.load('fashion_mnist_x_train.npy')
    y_train=np.load('fashion_mnist_y_train.npy')
    x_validation=np.load('fashion_mnist_x_validation.npy')
    y_validation=np.load('fashion_mnist_y_validation.npy')
    x_test=np.load('fashion_mnist_x_test.npy')
    y_test=np.load('fashion_mnist_y_test.npy')

if model=='alexnet':
    x_train = x_train.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_validation,x_train, x_test = color_preprocessing(x_validation,x_train, x_test)
else:
    x_train = x_train.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_validation, x_test,x_train=pre_processing(x_validation, x_test,x_train)
    
optimizer = SGD(lr=0.0008, momentum=0.9, decay=5e-4, nesterov=True)
model.compile(loss=keras.losses.MeanSquaredError(),optimizer=optimizer,metrics = ['categorical_accuracy'])
model.summary()

tmp_model = submodel(model)
tmp_model.summary()

value=[]
modify=[]

tmp = model.predict(x_validation)

for i in range(len(x_validation)):

    predict=np.argmax(tmp[i])
    y=y_validation[i]
    value.append(tmp[i][y])
    modify.append(0)

rs=np.zeros([len(x_validation),2048], dtype = float) 

for i in range(100,200):
    start = datetime.datetime.now()

    tf.keras.backend.clear_session()
    model_name='alexnet/alexnet_model.'
    if i<100:
        model_name=model_name+'0'+str(i)+'.h5'
    else:
        model_name=model_name+str(i)+'.h5'
    print(model_name)
    new_model=load_model(model_name)
    t = submodel(new_model)

    tmp2 = new_model.predict(x_validation)

    a = t.predict(x_validation)

    for j in range(len(x_validation)):  
        predict=np.argmax(tmp[j])
        y=y_validation[j]

        v=tmp2[j][y]
        if predict==y:
            if v>value[j]:
                rs[j]=a[j]
                value[j]=v
                modify[j]=1
        else:
            if np.argmax(tmp2[j])==y and v>value[j]:
                rs[j]=a[j]
                value[j]=v
                modify[j]=1
    elapsed = (datetime.datetime.now() - start)
    print("find Time used: ", elapsed)
c=0
for i in range(len(x_validation)):
    if modify[i]==1:
        c+=1
print(c)

np.save('cifar10_alexnet/{}_{}_{}_modify.npy'.format(dataset,model_name,choose_layer),modify)
np.save('cifar10_alexnet/{}_{}_{}_rs.npy'.format(dataset,model_name,choose_layer),rs)




