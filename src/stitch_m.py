# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:01:02 2016

@author: Luc
"""

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
from sklearn.metrics import confusion_matrix

dataset_dir = "../data/"
input_path = "D:/data/subset7/"
target_path = "D:/data/seg-lungs-LUNA16/seg-lungs-LUNA16/"
network_path = "../network.w"

#inputs = os.listdir(input_path)

filter_size = (3,3)
learning_rate = 0.000001
n_filters = 12 #64
n_dense = 1024 #4096
n_epochs = 50
n_batches = 1



def create_network():
    
    
    #input
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, 64, 64))
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
    
    #softmax = lasagne.layers.DenseLayer(output, num_units = 2, nonlinearity=lasagne.nonlinearities.softmax)
    #print lasagne.layers.get_output_shape(softmax)
    
    network = output
    return network
    


def training(network, train_X, train_Y, val_X, val_Y):
    
    #loss function

    lambda2=0.00001
    lr = 0.0001
    Y = T.fmatrix()
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
    test_prediction = lasagne.layers.get_output(network, inputs = X, deterministic = True)
    test_e_x = np.exp(test_prediction - test_prediction.max(axis =1, keepdims=True))
    test_out = (test_e_x / test_e_x.sum(axis=1, keepdims=True)).flatten(2)
    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_out, 0.0001, 0.9999), Y)
    test_loss = test_loss.mean() - l2_loss
    #acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), Y), dtype=theano.config.floatX)
    val_fn = theano.function([X, Y], [prediction, test_prediction, test_loss])#, acc])
    
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
        print "Run validation set"
        
        val_batches = 0
        val_loss = 0
        conf_matrix = np.zeros((2,2))
        #Validationset
        for inputs, targets in tqdm(Reader(meta_data = 'validation_set.stats')):
            predictions, test_prediction, test_loss = val_fn(inputs, targets)
            target_labels = [label.argmax() for label in targets]
            pred_labels = [label.argmax() for label in test_prediction]
            conf_matrix += confusion_matrix(target_labels, pred_labels, labels = [0,1])
            val_loss += test_loss
            val_batches += 1
            

                
                
            
        print "Validation loss: {}".format(val_loss/float(val_batches))
        print "True positives: {} \n False positives: {} \nFalse negatives {} \n True negatives {} (postive is lung, negative is background)".format(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1])
        
        
    print "Total runtime: " +str(time.time()-begin)
    
    network.save_weights_to(network_path)
    
    return 



    

if __name__ == '__main__':

    network = create_network()
    training(network, 0, 0, 0, 0)

    
    