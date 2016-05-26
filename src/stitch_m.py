# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:01:02 2016

@author: Luc
"""
import getpass

from read_data_m import Reader
from read_nodule_patches import NoduleReader
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import lasagne
from lasagne.layers import InverseLayer, batch_norm

import os
import time
import cPickle as pickle
import read_data
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


network_path = "../network.w"

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
n_epochs = 100
n_batches = 1



def create_network():
    
    
    #input
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, None, None))
    print lasagne.layers.get_output_shape(input_layer)
    
    #Conv 64
    conv64 = batch_norm(lasagne.layers.Conv2DLayer(input_layer, n_filters, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv64)
    
    #Max pool
    #pool0 = lasagne.layers.MaxPool2DLayer(conv64, pool_size=(2, 2))
    #print lasagne.layers.get_output_shape(pool0)
    
    #Conv x1 128
    conv128 = batch_norm(lasagne.layers.Conv2DLayer(conv64, n_filters*2, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv128)
    
    #Max pool
    pool1 = lasagne.layers.MaxPool2DLayer(conv128, pool_size=(2, 2))    
    print lasagne.layers.get_output_shape(pool1)
    
    #Conv x2 256
    conv256_0 = batch_norm(lasagne.layers.Conv2DLayer(pool1, n_filters*4, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv256_0)
    conv256_1 = batch_norm(lasagne.layers.Conv2DLayer(conv256_0, n_filters*4, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv256_1)
    
    #Max pool
    pool2 = lasagne.layers.MaxPool2DLayer(conv256_1, pool_size=(2, 2))        
    print lasagne.layers.get_output_shape(pool2)
    
    #Conv x2 512
    conv512_0 = batch_norm(lasagne.layers.Conv2DLayer(pool2, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv512_0)
    conv512_1 = batch_norm(lasagne.layers.Conv2DLayer(conv512_0, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv512_1)

    #Max pool
    pool3= lasagne.layers.MaxPool2DLayer(conv512_1, pool_size=(2, 2))    
    output_shape = lasagne.layers.get_output_shape(pool3)
    print lasagne.layers.get_output_shape(pool3)
    

    
    #Dense x2 4096    
    dropout0 = lasagne.layers.DropoutLayer(pool3, p=0.5) #check if dropout is needed 
    dense0 = batch_norm(lasagne.layers.Conv2DLayer(dropout0, n_dense, (4, 4), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    output_shape = lasagne.layers.get_output_shape(dense0)
    print output_shape
    
    #Dense x2 4096    
    dropout1 = lasagne.layers.DropoutLayer(dense0, p=0.5) #check if dropout is needed 
    dense1 = batch_norm(lasagne.layers.Conv2DLayer(dropout1, n_dense, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    output_shape = lasagne.layers.get_output_shape(dense1)
    print output_shape
    
    # a last layer of 2 neurons, that later on enter softmax
    output = batch_norm(lasagne.layers.Conv2DLayer(dense1, 2, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    output_shape = lasagne.layers.get_output_shape(output)
    print output_shape
    
    #softmax = lasagne.layers.DenseLayer(output, num_units = 2, nonlinearity=lasagne.nonlinearities.softmax)
    #print lasagne.layers.get_output_shape(softmax)
    
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
    lr = 0.001
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

    test_prediction = lasagne.layers.get_output(network, inputs = X, deterministic = True)
    test_e_x = np.exp(test_prediction - test_prediction.max(axis =1, keepdims=True))
    test_out = (test_e_x / test_e_x.sum(axis=1, keepdims=True)).flatten(2)
    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_out, 0.0001, 0.9999), Y)
    test_loss = loss.mean()

    val_fn = theano.function([X, Y], [test_prediction, test_loss])#, acc])
    
    begin = time.time()
    print "Start training" 
    for epoch in range(n_epochs):
        # In each epoch, we do a full pass over the training data...
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #for batch in iterate_minibatches(train_X, train_Y, 32, shuffle=True):
        print "epoch {}...".format(epoch)
        print "n_samples: {}".format(reader.n_samples)
        for inputs, targets in tqdm(NoduleReader()):           
            
            loss, l2_loss, prediction = train_fn(inputs, targets)
            targets = [label.argmax() for label in targets]
            """
            for i, label in enumerate(targets):
                print inputs.shape
                print  label
                plt.imshow(inputs[i, 0, :, :], cmap='gray')
                plt.show()

            """    

            
            train_err += loss
            train_batches+=1
            #print "Current subject: {} current slice: {}".format(reader.current, reader.current_slice)
            #print "n positve labels: {}".format(sum(targets))
            
            
            

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / float(train_batches)))
        #print "Run validation set"
        
        
        val_batches = 0
        val_loss = 0
        conf_matrix = np.zeros((2,2))
        #Validationset
        reader = NoduleReader(meta_data = 'validation_set.stat')
        for inputs, targets in tqdm(reader):
            test_prediction, test_loss = val_fn(inputs, targets)
            target_labels = [label.argmax() for label in targets]
            pred_labels = [label.argmax() for label in test_prediction]
            conf_matrix += confusion_matrix(target_labels, pred_labels, labels = [0,1])
            val_loss += test_loss
            val_batches += 1
            
                
        dice = dice_score(conf_matrix)
            
        print "Validation loss: {}".format(val_loss/float(val_batches))
        print "Dice score: {}".format(dice) 
        print "True positives: {} \n False negative: {} \nFalse positive {} \n True negatives {} (postive is lung, negative is background)".format(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1])

        
        print "save model..."
        np.savez('../networks/nodule_segmentation/network_epoch{}.npz'.format(epoch), *lasagne.layers.get_all_param_values(network))
        
        
    print "Total runtime: " +str(time.time()-begin)
    
def dice_score(conf_matrix):
    tp = conf_matrix[0][0]
    fn = conf_matrix[0][1]
    fp = conf_matrix[1][0]
    
    return tp*2. / (tp+fp+tp+fn)

if __name__ == '__main__':
    network = create_network()
    training(network, 0, 0, 0, 0)
