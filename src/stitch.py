# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:01:02 2016

@author: Luc
"""

from read_data import Reader
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import lasagne
from lasagne.layers import InverseLayer
import os
import time
import cPickle as pickle
import read_data
from tqdm import tqdm

dataset_dir = "../data/"
input_path = "D:/data/subset7/"
target_path = "D:/data/seg-lungs-LUNA16/seg-lungs-LUNA16/"

#inputs = os.listdir(input_path)

filter_size = (3,3)
learning_rate = 0.0001
n_filters = 12 #64
n_dense = 1024 #4096
n_epochs = 50
n_batches = 1


def prepare_trainings_data():
    n_samples = 10000
    train_X = np.zeros((n_samples*n_batches, 3, 32, 32), dtype = "float32")
    train_Y = np.zeros((n_samples*n_batches, 1), dtype="ubyte").flatten()
    
    for i in range(n_batches):
        f = open(os.path.join(dataset_dir, "data_batch_"+str(i+1)), "rb")
        cifar_batch = pickle.load(f)
        f.close()
        train_X[i*n_samples:(i+1)*n_samples] = (cifar_batch['data'].reshape(-1, 3, 32, 32)/255.).astype("float32")
        train_Y[i*n_samples:(i+1)*n_samples] = np.array(cifar_batch['labels'], dtype='ubyte')
        
    # validation set, batch 5    c
    f = open(os.path.join(dataset_dir, "data_batch_5"), "rb")
    cifar_batch_5 = pickle.load(f)
    f.close()
    val_X = (cifar_batch_5['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    val_Y = np.array(cifar_batch_5['labels'], dtype='ubyte')
    
    # labels
    f = open(os.path.join(dataset_dir, "batches.meta"), "rb")
    cifar_dict = pickle.load(f)
    label_to_names = {k:v for k, v in zip(range(10), cifar_dict['label_names'])}
    f.close()
    
    train_X = normalize_data(train_X)
    val_X = normalize_data(val_X)
    print("training set size: data = {}, labels = {}".format(train_X.shape, train_Y.shape))
    print("validation set size: data = {}, labels = {}".format(val_X.shape, val_Y.shape))

    return train_X, train_Y, val_X, val_Y

def normalize_data(data):
    
    mean = np.mean(data)
    std = np.mean(data)
    
    return (data-mean)/std


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            #data augmentation
            for idx in excerpt:
                if np.random.randint(2) > 0:                    
                    inputs[idx]=np.fliplr(inputs[idx])
        else:
            excerpt = slice(start_idx, start_idx + batchsize)      
        
        yield inputs[excerpt], targets[excerpt]

def create_network():
    inputs = T.tensor4('X')
    targets = T.tensor4('Y')
    
    #input
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var = inputs)
    print lasagne.layers.get_output_shape(input_layer)
    
    #Conv 64
    conv64 = lasagne.layers.Conv2DLayer(input_layer, n_filters, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    print lasagne.layers.get_output_shape(conv64)
    
    #Max pool
    #pool0 = lasagne.layers.MaxPool2DLayer(conv64, pool_size=(2, 2))
    #print lasagne.layers.get_output_shape(pool0)
    
    #Conv x1 128
    conv128 = lasagne.layers.Conv2DLayer(conv64, n_filters*2, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    print lasagne.layers.get_output_shape(conv128)
    
    #Max pool
    pool1 = lasagne.layers.MaxPool2DLayer(conv128, pool_size=(2, 2))    
    print lasagne.layers.get_output_shape(pool1)
    
    #Conv x2 256
    conv256_0 = lasagne.layers.Conv2DLayer(pool1, n_filters*4, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    print lasagne.layers.get_output_shape(conv256_0)
    conv256_1 = lasagne.layers.Conv2DLayer(conv256_0, n_filters*4, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    print lasagne.layers.get_output_shape(conv256_1)
    
    #Max pool
    pool2 = lasagne.layers.MaxPool2DLayer(conv256_1, pool_size=(2, 2))        
    print lasagne.layers.get_output_shape(pool2)
    
    #Conv x2 512
    conv512_0 = lasagne.layers.Conv2DLayer(pool2, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    print lasagne.layers.get_output_shape(conv512_0)
    conv512_1 = lasagne.layers.Conv2DLayer(conv512_0, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    print lasagne.layers.get_output_shape(conv512_1)

    #Max pool
    pool3= lasagne.layers.MaxPool2DLayer(conv512_1, pool_size=(2, 2))    
    output_shape = lasagne.layers.get_output_shape(pool3)
    print lasagne.layers.get_output_shape(pool3)
    
    #Conv x2 512
    #conv512_2 = lasagne.layers.Conv2DLayer(pool3, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify)
    #print lasagne.layers.get_output_shape(conv512_2)
    #conv512_3= lasagne.layers.Conv2DLayer(conv512_2, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify)
    #print lasagne.layers.get_output_shape(conv512_3)
    
    #Max pool
    #pool4 = lasagne.layers.MaxPool2DLayer(conv512_3, pool_size=(2, 2))   
    #output_shape = lasagne.layers.get_output_shape(pool4)
    #print output_shape
    
    #Dense x2 4096    
    dropout0 = lasagne.layers.DropoutLayer(pool3, p=0.5) #check if dropout is needed 
    dense0 = lasagne.layers.Conv2DLayer(dropout0, n_dense, (4, 4), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    output_shape = lasagne.layers.get_output_shape(dense0)
    print output_shape
    
    #Dense x2 4096    
    dropout1 = lasagne.layers.DropoutLayer(dense0, p=0.5) #check if dropout is needed 
    dense1 = lasagne.layers.Conv2DLayer(dropout1, n_dense, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    output_shape = lasagne.layers.get_output_shape(dense1)
    print output_shape
    
    output = lasagne.layers.Conv2DLayer(dense1, 1, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    output_shape = lasagne.layers.get_output_shape(output)
    print output_shape
    
    
    network = output
    return inputs, targets, network
    


def training(inputs, targets, network, train_X, train_Y, val_X, val_Y):
    
    #loss function
    prediction = lasagne.layers.get_output(network) + 0.000001
    loss = lasagne.objectives.binary_crossentropy(prediction, targets) + 0.01 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss = loss.mean()
    loss = T.clip(loss, -1, 1)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)

    """
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets)
    test_loss = test_loss.mean()
    acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets), dtype=theano.config.floatX)
    """
    #Train anv validation functions    
    train_fn = theano.function([inputs, targets], [loss, prediction], updates=updates)
    #val_fn = theano.function([inputs, targets], [test_prediction, test_loss, acc])
    
    begin = time.time()
    print "Start training" 
    # The number of epochs specifies the number of passes over the whole training data
    curves = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(n_epochs):
        # In each epoch, we do a full pass over the training data...
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #for batch in iterate_minibatches(train_X, train_Y, 32, shuffle=True):
        print "epoch {}...".format(epoch)
        for inputs, targets in tqdm(Reader()):
            loss, prediction = train_fn(inputs, targets)
            train_err += loss
            train_batches+=1

    
        # ...and a full pass over the validation data
        """       
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_X, val_Y, 500, shuffle=False):
            inputs, targets = batch
            preds, err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        """
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / float(train_batches)))
        #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  validation accuracy:\t\t{:.2f} %".format(
        #    val_acc / val_batches * 100))
        #curves['train_loss'].append(train_err / train_batches)
        #curves['val_loss'].append(val_err / val_batches)
        #curves['val_acc'].append(val_acc / val_batches)

    print "Total runtime: " +str(time.time()-begin)
    
    return #curves, #val_fn 
    
def save_result(curves, file_path = '../plots/', name = "vdcn"):
    plt.plot(zip(curves['train_loss'], curves['val_loss']));
    plt.savefig(file_path + name + 'loss.png')
    plt.clf()
    plt.plot(curves['val_acc']);
    plt.savefig(file_path + name + 'accuracy.png')
    plt.clf()
    print "saved plots"     


if __name__ == '__main__':
    #train_x, train_y, val_x, val_y = prepare_trainings_data()
    inputs, targets, network = create_network()
    curves, val_fn = training(inputs, targets, network, 0, 0, 0, 0)
    #save_result(curves)
    
    