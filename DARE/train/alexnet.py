import tensorflow as tf
#import keras
from tensorflow import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,Reshape,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import optimizers

batch_size   = 128
epochs       = 100
iterations   = 196
np.set_printoptions(threshold=np.inf)
def scheduler(epoch):
    if epoch < 80:
        return 0.01
    if epoch < 160:
        return 0.005
    return 0.001
    
def color_preprocessing(x_validation,x_train,x_test):
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_validation = x_validation.astype('float32')/255.
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_validation -= x_train_mean
  #  mean = [125.307, 122.95, 113.865]
   # std  = np.std(x_train, axis=0)
 #   for i in range(3):
  #      x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
   #     x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_validation,x_train, x_test
    
x_validation=np.load('/home/youhanmo/202/zyy/new_adv/adv/cifar10_vgg16/cifar10_x_validation.npy')
y_validation=np.load('/home/youhanmo/202/zyy/new_adv/adv/cifar10_vgg16/cifar10_y_validation.npy')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=np.load('/home/youhanmo/202/zyy/new_adv/adv/cifar10_vgg16/cifar10_x_train.npy')
y_train=np.load('/home/youhanmo/202/zyy/new_adv/adv/cifar10_vgg16/cifar10_y_train.npy')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
x_validation = x_validation.astype('float32')
x_validation /= 255.
y_test = keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)
y_validation = keras.utils.to_categorical(y_validation, 10)
x_train_mean = np.mean(x_train, axis=0)
x_validation -= x_train_mean
x_test -= x_train_mean
x_train -= x_train_mean

x_train=x_train.reshape(-1,32,32,3)
x_test=x_test.reshape(-1,32,32,3)

def AlexNet8():
    inputs = Input(shape=(32,32,3))
   # inputs = Input(shape=(28,28,1))
    x=inputs
    x = Conv2D(filters=96, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')(x)

    x =  Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')(x)
    x =  MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=y)
    return model
    
def vgg16(input_shape):
    input_tensor=Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    print(x.shape)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    print(x.shape)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    print(x.shape)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    print(x.shape)
    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # print(x.shape)
 
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    model = Model(inputs=input_tensor, outputs=x, name='VGG16')
    return model

def vgg19(input_shape):
    input_tensor=Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((1, 2), strides=(2, 2), name='block1_pool')(x)
    print(x.shape)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 1), strides=(2, 2), name='block2_pool')(x)
    print(x.shape)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((1, 2), strides=(2, 2), name='block3_pool')(x)
    print(x.shape)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 1), strides=(2, 2), name='block4_pool')(x)
    print(x.shape)
     # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    print(x.shape)
 
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    model = Model(inputs=input_tensor, outputs=x, name='VGG19')
    return model
model = AlexNet8()

#model = vgg19(input_shape = (28, 28, 1))
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
#tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
save_dir = os.path.join(os.getcwd(), 'for_heat_map')
model_name = 'vgg16_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)
cbks = [checkpoint,change_lr]

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
model.save('cifar10_alexnet.h5')

