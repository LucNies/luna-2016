# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:01:02 2016

@author: Luc
"""
import getpass

from read_data_m import Reader
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
    
    
    #input
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, None, None))
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
    
    # a last layer of 2 neurons, that later on enter softmax
    output = lasagne.layers.Conv2DLayer(dense1, 2, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal())
    output_shape = lasagne.layers.get_output_shape(output)
    print output_shape    
    
    network = output
    return network


def shift_matrix(matrix, amount):
    i,j = amount
    N,M = matrix.shape[2:]  # Required for a 0-shift
    matrix_ = np.zeros_like(matrix)
    matrix_[:, :, i:, j:] = matrix[:, :, :N-i, :M-j]
    return matrix_


def shift(inputs, network):
    indims = inputs.shape
    outdims = lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network), indims)
    outdims = outdims[-1]
    n_shifts = (indims[2] / float(outdims[2]),
                indims[3] / float(outdims[3]))
    n_shifts = (int(np.ceil(n_shifts[0])), int(np.ceil(n_shifts[1])))

    for i in range(n_shifts[0]):
        for j in range(n_shifts[1]):
            yield shift_matrix(inputs, (i,j))


def training(network, train_X, train_Y, val_X, val_Y):
    
    #loss function

    lambda2=0.00001
    lr = 0.0001
    Y = T.imatrix()
    ftensor4 = T.TensorType('float32', (False,)*4)
    X = ftensor4()
    prediction = lasagne.layers.get_output(network, inputs = X)
    e_x = np.exp(prediction - prediction.max(axis=1, keepdims=True))
    out = (e_x / e_x.sum(axis=1, keepdims=True)).flatten(2)
    loss = lasagne.objectives.categorical_crossentropy(T.clip(out, 0.0001, 0.9999), Y)
    l2_loss =  lambda2 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss = loss.mean() + l2_loss
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
    train_fn = theano.function([X, Y], [loss, l2_loss, prediction], updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    """
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets)
    test_loss = test_loss.mean()
    acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets), dtype=theano.config.floatX)
    """
    #Train anv validation functions    
    #train_fn = theano.function([inputs, targets], [loss, prediction], updates=updates)
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
            loss, l2_loss, prediction = train_fn(inputs, targets)
            train_err += loss
            train_batches+=1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / float(train_batches)))

    print "Total runtime: " +str(time.time()-begin)
    
    return


def stitch(predictions, prediction, i, width, height):
    patch_cols = slice(i, predictions.shape[1], width)
    patch_rows = slice(i, predictions.shape[2], height)
    predictions[:, patch_cols, patch_rows] = prediction
    return predictions


def accuracy(predictions, targets):
    good = (predictions == targets).sum()
    total = targets.sum()
    return good/float(total)


def test(network):
    reader = Reader(patch_shape=(512, 512))
    for inputs, _ in tqdm(reader):
        _, targets = reader.load_itk_images(*reader.get_locations())
        targets = targets >= 3
        predictions = np.zeros_like(targets)
        width = targets.shape[1] / lasagne.layers.get_output_shape(network, inputs.shape)[2]
        height = targets.shape[2] / lasagne.layers.get_output_shape(network, inputs.shape)[3]

        i = 0
        for inputs_ in shift(inputs, network):
            prediction = lasagne.layers.get_output(network, inputs_)
            predictions = stitch(predictions, prediction, i, width, height)
            i+=1

        print "Accuracy: {}".format(accuracy(predictions, targets))
    return

if __name__ == '__main__':
    network = create_network()
    training(network, 0, 0, 0, 0)
    test(network)

