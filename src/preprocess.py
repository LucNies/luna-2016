# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 22:02:05 2016

@author: Luc
"""
from __future__ import division
from tqdm import tqdm
import numpy as np
import os
import SimpleITK as sitk
import math
import pickle

VERSION = 1

def preprocess(file_path = 'D:/data/subset', out_path='D:/data/preprocessed/dataset.p'):
    
    i=9
    full_path = file_path + str(i) + '/'
    file_names = os.listdir(full_path)
    full_names = np.chararray(len(file_names)/2, itemsize = len(full_path + file_names[0]))
        
    
    print 'Creating file list'
    for i, name in enumerate(file_names[0::2]):
       full_names[i] = full_path + name
    print 'Done'
    
    print 'Calculating mean and std of all images...'
    mean, std = calc_stat(full_names)
    print 'Done... mean: {}, std: {}'.format(mean,std) 
    
    print 'saving metadata...'
    metadata = {}
    metadata['version'] = VERSION
    metadata['mean'] = mean
    metadata['std'] = std
    metadata['n_samples'] = len(full_names)
    metadata['file_names'] = full_names
    pickle.dump(metadata, open('image_stats.stat', 'wb'))
    print 'done saving'

    dataset = {}
    """
    if os.path.exists(out_path):
    with open(out_path, 'rb') as rfp:
        dataset = pickle.load(rfp)    
    """
    print "Normalizing images..."
    for i, file_name in tqdm(enumerate(full_names[:10])):
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_name)) 
        image = (image - mean)/std
        split = file_name.split('.')
        dataset[split[len(split)-2]] = {'image':image}
        """
        if (i+1)%10 == 0:
            with open(out_path, 'rb') as read:
                saved_data = pickle.load(read)
            save_data.update()
        """
    print "Normalizing done."
    
    print "Saving dataset... (sry, no progressbar this time... takes awefull long though)"
    with open(out_path, 'wb') as write:
        pickle.dump(dataset, write)
    print 'Saving done.'
    

    
    print "Done preprocessing"

#calc mean and std over all images
def calc_stat(file_names):
    
    n = 0
    total_mean = 0.0    
    M2 = 0.0    
    
    for file_name in tqdm(file_names):
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_name)) 
        mean = image.mean()
        n += 1
        delta =  mean - total_mean
        mean += delta/n
        M2 += delta*(mean - total_mean)
    
    return mean, math.sqrt(M2/(n-1))


def get_subject_name(file_name):
    
    split = file_name.split('.')
    
    return split[len(split)-2]
    

if __name__ == '__main__' :
    preprocess()