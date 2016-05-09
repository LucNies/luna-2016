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
import getpass

if getpass.getuser() == 'harmen':
    dataset_dir = "../data/"
    input_path = "../data/subset0/"
    target_path = "../data/seg-lungs-LUNA16/seg-lungs-LUNA16/"
else:
    dataset_dir = "../data/"
    input_path = "D:/data/subset7/"
    target_path = "D:/data/seg-lungs-LUNA16/seg-lungs-LUNA16/"

#inputs = os.listdir(input_path)

filter_size = (3,3)
learning_rate = 0.000001
n_filters = 12 #64
n_dense = 1024 #4096
n_epochs = 50
n_batches = 1



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


def shift_matrix(matrix, amount):
    i,j = amount
    N,M = matrix.shape()  # Required for a 0-shift
    matrix_ = np.zeros_like(matrix)
    matrix_[i:, j:] = matrix[:N-i, :M-j]
    return matrix_


def shift(inputs, targets, network):
    indims = inputs.shape()
    outdims = network.get_output_shape()

    n_shifts = (indims[0] / float(outdims[0]),
                indims[1] / float(outdims[1]))
    assert(n_shifts[0].is_integer() and n_shifts[1].is_integer())
    n_shifts = (int(n_shifts[0]), int(n_shifts[1]))

    for i in range(n_shifts[0]):
        for j in range(n_shifts[1]):
            input_ = shift_matrix(input, (i,j))
            if targets is not None:
                targets_ = shift_matrix(targets, (i,j))
                yield input_, targets_
            else:
                yield input_




def training(inputs, targets, network, train_X, train_Y, val_X, val_Y):
    
    #loss function
    prediction = lasagne.layers.get_output(network) + 0.00001
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
    for epoch in range(n_epochs):
        # In each epoch, we do a full pass over the training data...
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #for batch in iterate_minibatches(train_X, train_Y, 32, shuffle=True):
        print "epoch {}...".format(epoch)
        for inputs, targets in tqdm(Reader()):
            for inputs_, targets_ in shift(inputs, targets, network):
                loss, prediction = train_fn(inputs_, targets_)
                train_err += loss
            train_batches+=1


        
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / float(train_batches)))

    print "Total runtime: " +str(time.time()-begin)
    
    return 
    

if __name__ == '__main__':

    inputs, targets, network = create_network()
    training(inputs, targets, network, 0, 0, 0, 0)


    
    