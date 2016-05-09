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
subsets = range(1) # not subset9, use that as testset

def preprocess(file_path = 'D:/data/subset'):
    full_names = []
    print 'Creating file list'
    print 'Processing {} subsets, is this ok?'.format(len(subsets))
    for i in subsets:
        full_path = file_path + str(i)
        file_names = os.listdir(full_path)
        file_names = [fn for fn in file_names if ".mhd" in fn]
        for name in file_names[0::2]:
           full_names.append(os.path.join(full_path, name))

    print 'Done, {} filenames found'.format(len(full_names))
    
    print 'Calculating mean and std of all images...'
    mean, std, n_slices = calc_stat(full_names)
    print 'Done... mean: {}, std: {}, total of: {}'.format(mean, std, n_slices) 
    
    print 'saving metadata...'
    metadata = {}
    metadata['version'] = VERSION
    metadata['mean'] = mean
    metadata['std'] = std
    metadata['n_samples'] = len(full_names)
    metadata['file_names'] = full_names
    metadata['n_slices'] = n_slices
    with open('image_stats.stat', 'wb') as write:
        pickle.dump(metadata, write)
    print 'done saving'

   

    
    print "Done preprocessing"

#calc mean and std over all images, and total number of slices
def calc_stat(file_names):
    
    n = 0
    total_mean = 0.0    
    M2 = 0.0    
    n_slices = 0
    
    for file_name in tqdm(file_names):
        full_name = os.path.abspath(file_name)
        image = sitk.GetArrayFromImage(sitk.ReadImage(full_name))
        n_slices += image.shape[0]
        mean = image.mean()
        n += 1
        delta =  mean - total_mean
        mean += delta/n
        M2 += delta*(mean - total_mean)
    
    return mean, math.sqrt(M2/(n-1)), n_slices


def get_subject_name(file_name):
    
    split = file_name.split('.')
    
    return split[len(split)-2]
    

if __name__ == '__main__' :
    preprocess(os.path.join("..", "data", "subset"))