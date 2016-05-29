# -*- coding: utf-8 -*-
"""
Created on Wed May 18 00:04:46 2016

@author: The Mountain
"""

from stitch_m import create_network
from read_images import ImageReader
from tqdm import tqdm
import lasagne
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


def dice_score(p,t):
    t = np.asarray(t)
    return np.sum(p[t == 1]) * 2.0 / (np.sum(p) + np.sum(t))

def stitch(predictions, prediction, i, width, height):
    y = i / width
    x = i % width

    patch_cols = slice(x, predictions.shape[1], width)
    patch_rows = slice(y, predictions.shape[2], height)

    predictions[:, patch_rows, patch_cols] = prediction
    return predictions

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

def load_network(network_path = '../networks/trained_segmentation/network_epoch24.npz'):
    network = create_network()
    with np.load(network_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        
    lasagne.layers.set_all_param_values(network, param_values)
    
    return network
    
def fix(predictions, width, height):
    fixed = np.zeros_like(predictions)
    for s in range(predictions.shape[0]):
        for x in range(0, 513, width):
            for y in range(0, 513, height):
                fixed[s, x:x+width, y:y+width] = np.rot90(predictions[s, x:x+width, y:y+width], k=2)
                
    return fixed

def test():
    network = load_network()
    
    reader = ImageReader()
    for inputs, targets in tqdm(reader):
        predictions = np.zeros((targets.shape[0], targets.shape[1]+1, targets.shape[2]+1))
        
        inputs = inputs.reshape(-1, 1, 512, 512)
        
        width = targets.shape[1] / lasagne.layers.get_output_shape(network, inputs.shape)[2]
        height = targets.shape[2] / lasagne.layers.get_output_shape(network, inputs.shape)[3]

        X = T.tensor4('X')
        prediction = lasagne.layers.get_output(network, inputs=X)
        e_x = T.exp(prediction - prediction.max(axis=1, keepdims=True))
        out = (e_x / e_x.sum(axis=1, keepdims=True))
        out = T.argmax(out, axis=1)
        test_fn = theano.function([X], out)

        i = 0
        for inputs_ in tqdm(shift(inputs, network)):
            j = 0
            
            subject_predictions = np.zeros((targets.shape[0], 57, 57))
            
            for slice_ in inputs_:
                slice_ = slice_.reshape(1, 1, 512, 512)
                prediction = test_fn(slice_)
                
                subject_predictions[j] = prediction[0]

                j += 1

            #plt.imshow(subject_predictions[60], cmap='gray')
            #plt.show()
            predictions = stitch(predictions, subject_predictions, i, width+1, height+1)
            i += 1
            
            # Dirty fix
            fixed_predictions = fix(predictions, width+1, height+1)
            

            plt.imshow(fixed_predictions[60, :, :], cmap='gray')
            plt.savefig(str(i) + '.png')
            
        predictions = fixed_predictions[:, :512, :512]

        print "Accuracy: {}".format(dice_score(fixed_predictions, targets))
    return

if __name__ == '__main__':
    test()
