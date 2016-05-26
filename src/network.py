# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:16:15 2016

@author: The Mountain
"""
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, batch_norm 
import lasagne
import numpy as np


def create_network(n_filters = 12, n_dense = 1024, filter_size = (3,3)):
    
    
    #input
    input_layer = InputLayer(shape=(None, 1, None, None))
    print lasagne.layers.get_output_shape(input_layer)
    
    #Conv 12
    conv12 = batch_norm(Conv2DLayer(input_layer, n_filters, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv12)
    
    #Max pool
    #pool0 = lasagne.layers.MaxPool2DLayer(conv12, pool_size=(2, 2))
    #print lasagne.layers.get_output_shape(pool0)
    
    #Conv x1 24
    conv24 = batch_norm(Conv2DLayer(conv12, n_filters*2, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv24)
    
    #Max pool
    pool1 = MaxPool2DLayer(conv24, pool_size=(2, 2))    
    print lasagne.layers.get_output_shape(pool1)
    
    #Conv x2 48
    conv48_0 = batch_norm(Conv2DLayer(pool1, n_filters*4, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv48_0)
    conv48_1 = batch_norm(Conv2DLayer(conv48_0, n_filters*4, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv48_1)
    
    #Max pool
    pool2 = MaxPool2DLayer(conv48_1, pool_size=(2, 2))        
    print lasagne.layers.get_output_shape(pool2)
    
    #Conv x2 96
    conv96_0 = batch_norm(Conv2DLayer(pool2, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv96_0)
    conv96_1 = batch_norm(Conv2DLayer(conv96_0, n_filters*8, filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    print lasagne.layers.get_output_shape(conv96_1)

    #Max pool
    pool3= MaxPool2DLayer(conv96_1, pool_size=(2, 2))    
    output_shape = lasagne.layers.get_output_shape(pool3)
    print lasagne.layers.get_output_shape(pool3)
    

    
    #Dense x2 1024    
    dropout0 = DropoutLayer(pool3, p=0.5) #check if dropout is needed 
    dense0 = batch_norm(Conv2DLayer(dropout0, n_dense, (4, 4), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    output_shape = lasagne.layers.get_output_shape(dense0)
    print output_shape
    
    #Dense x2 1024    
    dropout1 = DropoutLayer(dense0, p=0.5) #check if dropout is needed 
    dense1 = batch_norm(Conv2DLayer(dropout1, n_dense, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    output_shape = lasagne.layers.get_output_shape(dense1)
    print output_shape
    
    # a last layer of 2 neurons, that later on enter softmax
    output = batch_norm(Conv2DLayer(dense1, 2, (1, 1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal()))
    output_shape = lasagne.layers.get_output_shape(output)
    print output_shape
    
    
    network = output
    return network

def load_network(network_path = '../networks/nodule_segmentation/network_epoch5.npz'):

    network = create_network()
    with np.load(network_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        
    lasagne.layers.set_all_param_values(network, param_values)
    
    return network
    