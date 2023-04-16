#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import keras
from keras import Model,Input
from keras.models import load_model
from keras.layers import Activation,Flatten
import math
import numpy as np
import pandas as pd
import foolbox
from tqdm import tqdm
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import scipy
import sys,os
# import SVNH_DatasetUtil
import itertools
sys.path.append('./fashion-mnist/utils')
# import mnist_reader
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import datetime

from keras.applications import inception_v3,vgg19,resnet50
import traceback
import argparse


def adv_func(x,y,model_path='./model/model_mnist.hdf5',dataset='mnist',attack='fgsm',mean=0):
    keras.backend.set_learning_phase(0)
    model=load_model(model_path)

    foolmodel = foolbox.models.KerasModel(model, bounds=(0, 255))
    if attack=='cw':

        attack=foolbox.attacks.CarliniWagnerL2Attack(foolmodel)
    elif attack=='fgsm':
        # FGSM
        attack=foolbox.attacks.GradientSignAttack(foolmodel)
    elif attack=='bim':
        # BIM
        metric = foolbox.distances.MAE
        attack=foolbox.attacks.L1BasicIterativeAttack(foolmodel)
    elif attack=='jsma':
        # JSMA
        attack=foolbox.attacks.SaliencyMapAttack(foolmodel)
    elif attack=='pgd':
        # PGD
        attack=foolbox.attacks.ProjectedGradientDescentAttack(foolmodel)
     
    result=[]
    if dataset=='mnist':
        w,h=28,28
    elif dataset=='cifar10' or dataset=='cifar100':
        w,h=32,32
    elif dataset=='imagenet':
        w,h=224,224
    else:
        return False

    y_list = []
    for b in range(x.shape[0]):
        y_list.append(y)
    print(x.shape)
    y_list = np.array(y_list)
    print(y_list.shape)
    print(len(x))
    print(len(y_list))

  #  adv = attack(x, y_list,theta=0.01,max_perturbations_per_pixel=30)
    adv = attack(x, y_list )
  #  adv = attack(x, y_list ,confidence =5)
    return adv


def generate_cifar_sample(label,attack):

    X_train=np.load('cifar10_x_train.npy')
    print(X_train.shape)
    Y_train=np.load('cifar10_y_train.npy')
    print(Y_train.shape)
    image_org=X_train[Y_train.reshape(-1)==label]
    adv=adv_func(image_org,label,model_path='cifar10/alexnet_model.200.h5',dataset='cifar10',attack=attack)

    return adv


def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels

def pre_processing(train_images, test_images):
    images = np.concatenate((train_images, test_images), axis = 0)
    mean, std = get_mean_std(images)

    for i in range(test_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images    

    
def generate_fashion_sample(label,attack):
    X_train=np.load('fashion_mnist_x_train.npy')
    Y_train=np.load('fashion_mnist_y_train.npy')
    X_train = X_train.astype('float32').reshape(-1,28,28,1)

    X_train /= 255
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean


    image_org=X_train[Y_train==label]
    

    adv=adv_func(image_org,label,model_path='fashion_mnist/alexnet_model.200.h5',dataset='mnist',attack=attack)

    return adv

def generate_svhn_sample(label,attack):

    X_train=np.load('svhn_x_train.npy')
    Y_train=np.load('svhn_y_train.npy')
    Y_train-=1
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_train/=255.0
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean
    
    image_org=X_train[Y_train.reshape(-1)==label]
    
    adv=adv_func(image_org,label,model_path='svhn/alexnet_model.200.h5',dataset='cifar10',attack=attack)

    return adv


def generate_adv_sample(dataset,attack):
    if dataset=='mnist':
        sample_func=generate_mnist_sample
    elif dataset=='svhn':
        sample_func=generate_svhn_sample
    elif dataset=='fashion':
        sample_func=generate_fashion_sample
    elif dataset=='cifar10':
        sample_func=generate_cifar_sample

    else:
        print('erro')
        return
    image=[]
    label=[]
    for i in range(10):
        print(i)
        start = datetime.datetime.now()
        adv=sample_func(label=i,attack=attack)
        print(adv)
        if len(adv)==0:
            continue
        # print(adv)
        temp_image=adv
        temp_label=i*np.ones(len(adv))
        image.append(temp_image.copy())
        label.append(temp_label.copy())
        elapsed = (datetime.datetime.now() - start)
        print("Time used: ", elapsed)
    image=np.concatenate(image,axis=0)
    label=np.concatenate(label,axis=0)
 
    np.save('{}_{}_alexnet_image'.format(attack,dataset),image)
    np.save('{}_{}_alexnet_label'.format(attack,dataset),label)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", "-count", help="", type=int)
    args = parser.parse_args()
    start = datetime.datetime.now()
    '''
    svhn fashion cifar10
    cw fgsm bim jsma
    '''

    generate_adv_sample('cifar10', 'bim')
    elapsed = (datetime.datetime.now() - start)
    print("Time used: ", elapsed)

