# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:21:02 2016

@author: The Mountain
"""

from util import dice_score
import theano
import theano.tensor as T
import lasagne
import numpy as np
from tqdm import tqdm
from read_data_m import Reader
import time
from sklearn.metrics import confusion_matrix
from network import load_network, create_network

filter_size = (3,3)
learning_rate = 0.000001
n_filters = 12 #64
n_dense = 1024 #4096
n_epochs = 100
n_batches = 1



def training(network):
    
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
        reader = Reader()
        print "n_samples: {}".format(reader.n_samples)
        for inputs, targets in tqdm(reader):           
            
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
        reader = Reader(meta_data = 'validation_set.stat')
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
        print "True positives: {} \n False negative: {} \nFalse positive {} \n True negatives {} (postive is lung, negative is background)".format(conf_matrix[1][1], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[0][0])

        
        print "save model..."
        np.savez('../networks/lung_segmentation/network_epoch{}.npz'.format(epoch), *lasagne.layers.get_all_param_values(network))
        
        
    print "Total runtime: " +str(time.time()-begin)
    
if __name__ == "__main__":
    #network = load_network()
    network = create_network()
    training(network)