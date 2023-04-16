import os
import time
import argparse

from keras.models import load_model
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, \
    TensorBoard, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger


from utils import get_model, get_cifar_gen, find_lr, get_best_checkpoint, save_train_images
from params import *


def new_loss_fuction(y_actual,y_predicted):
    t=10
    y_t=K.softmax(y_predicted/t)
    y_a=K.softmax(y_actual)
    
    return K.categorical_crossentropy(y_a, y_t)

if __name__ == "__main__":
    cur_time = '-'.join(time.ctime().split(' ')).replace(':', '-')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name', default=model_name)
    parser.add_argument('--gpu', type=str, help='visible devices', default=visible_gpu)
    parser.add_argument('--batch_size', type=int, help='batch size for dataloader', default=batch_size)
    parser.add_argument('--epochs', type=int, help='num epochs', default=epochs)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=lr)
    parser.add_argument('--checkpoint', type=str, help='train from a previous model', default=None)
    parser.add_argument('--early_stop', type=bool, help='apply early stop', default=False)
    parser.add_argument('--continue_train', type=bool, help='continue train from previous best', default=False)
    parser.add_argument('--optimizer', type=str, help='choose optimizer', default='sgd')
    args = parser.parse_args()

    print('####################################')
    print(time.ctime())
    print('hyper param selection')
    print('model name:', args.model)
    print('batch size:', args.batch_size)
    print('epoch num: ', args.epochs)
    print('initial lr:', args.lr)
    print('optimizer: ', args.optimizer)
    print('####################################')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_path = os.path.join(base_path, 'dis_experiments', args.model)
    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, 'logs'))
        os.makedirs(os.path.join(model_path, 'checkpoints'))

    model = get_model(args.model)
   # plot_model(model, to_file=os.path.join(model_path, args.model + '.png'), show_shapes=True)

    # callbacks
    # checkpoint callback
    model_names = os.path.join(model_path, 'checkpoints',
                               cur_time + '.model.{epoch:03d}.hdf5')
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc',
                                       verbose=1, save_best_only=False, 
                                       save_weights_only=False)
    # lr callback
    lr_scheduler = LearningRateScheduler(find_lr, verbose=1)
    # tensor board callback
    tb_dir = os.path.join(model_path, 'logs', cur_time)
    os.makedirs(tb_dir)
    tensor_board = TensorBoard(log_dir=tb_dir,
                               histogram_freq=0, write_graph=True, write_images=True)
    csv_logger = CSVLogger(os.path.join(tb_dir, args.model + '_logger.csv'))

    callbacks = [model_checkpoint, tensor_board, lr_scheduler, csv_logger]

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = SGD(lr=args.lr, momentum=0.9, decay=5e-4, nesterov=True)
    else:
        optimizer = Adam(lr=args.lr)
    model.compile(optimizer=optimizer, metrics=['accuracy'],
                  loss=new_loss_fuction)

    # more options about lr
    if args.early_stop:
        early_stop = EarlyStopping('val_loss', patience=patience)
        callbacks.append(early_stop)
    if args.checkpoint is not None:  # train from a checkpoint
        model = load_model(os.path.join(model_path, 'checkpoints', args.checkpoint))
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=int(patience / 2), verbose=1)
        callbacks.append(reduce_lr)
    if args.continue_train:
        cpt = get_best_checkpoint(args.model)
        if cpt:
            model = load_model(os.path.join(model_path, 'checkpoints', cpt))
            if lr != args.lr:
                cur_lr = args.lr
            else:
                cur_lr = 1e-4
            if args.optimizer == 'sgd':
                optimizer = SGD(lr=cur_lr, momentum=0.9, decay=5e-4, nesterov=True)
            else:
                optimizer = Adam(lr=cur_lr)
            model.compile(optimizer=optimizer, metrics=['accuracy'],
                          loss='categorical_crossentropy')
            reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=int(patience / 2), verbose=1)
            callbacks = [model_checkpoint, tensor_board, reduce_lr]

    cifar_gen, cifar_test_gen = get_cifar_gen()
    history = model.fit_generator(cifar_gen,
                                  epochs=epochs,
                                  validation_data=cifar_test_gen,
                                  callbacks=callbacks,
                                  verbose=2)
 #   model.save('vgg16_cifar10_50000.h5')
    #save_train_images(history, tb_dir)
