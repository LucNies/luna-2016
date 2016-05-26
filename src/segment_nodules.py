# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:22:29 2016

@author: The Mountain
"""

import segment_lungs as segment
import numpy as np
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from tqdm import tqdm
from read_images import ImageReader
import os



def test(network_path = '../networks/nodule_segmentation/network_epoch49.npz'):
    network = segment.load_network(network_path)

    reader = ImageReader()    

    for inputs, lung_mask, subject_name in tqdm(reader):
        predictions = np.zeros((inputs.shape[0], inputs.shape[1]+1, lung_mask.shape[2]+1))
        
        inputs = inputs.reshape(-1, 1, 512, 512)
        
        width = lung_mask.shape[1] / lasagne.layers.get_output_shape(network, inputs.shape)[2]
        height = lung_mask.shape[2] / lasagne.layers.get_output_shape(network, inputs.shape)[3]

        X = T.tensor4('X')
        prediction = lasagne.layers.get_output(network, inputs=X)
        e_x = T.exp(prediction - prediction.max(axis=1, keepdims=True))
        out = (e_x / e_x.sum(axis=1, keepdims=True))
        out = T.argmax(out, axis=1)
        test_fn = theano.function([X], out)
        
        i = 0
        for inputs_ in tqdm(segment.shift(inputs, network)):
            j = 0
            
            subject_predictions = np.zeros((inputs.shape[0], 57, 57))
            
            for slice_ in inputs_:
                slice_ = slice_.reshape(1, 1, 512, 512)
                prediction = test_fn(slice_)
                
                subject_predictions[j] = prediction[0]

                j += 1


            predictions = segment.stitch(predictions, subject_predictions, i, width+1, height+1)
            i += 1
            
            # Dirty fix
            fixed_predictions = segment.fix(predictions, width+1, height+1)
            fixed_predictions = fixed_predictions[:, :512, :512]*lung_mask

            #plt.imshow(fixed_predictions[60, :, :], cmap='gray')
            #plt.show()
            #plt.savefig(str(i) + '.png')
            
        #predictions = fixed_predictions[:, :512, :512]
        
        np.savez_compressed('../segmentations/nodules/'+subject_name, fixed_predictions)

        # Load: np.load(os.path.join(data_dir, 'candidates', f + '.npz'))['arr_0']
        #print "Accuracy: {}".format(dice_score(predictions, targets))
    return


if __name__ == '__main__':
    test()
    