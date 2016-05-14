# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:39:52 2016

@author: Luc-squad
"""
from __future__ import division
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import csv
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import util
import preprocess
import pickle
<<<<<<< HEAD
from sklearn.feature_extraction.image import extract_patches_2d

patch_size = 64
=======
import getpass
>>>>>>> eee58529fa4d1d66175b1b52e541fc715a7d363d

if getpass.getuser() == 'harmen':
    lbl_path = os.path.join("..", "data", "seg-lungs-LUNA16")
else:
    lbl_path = 'D:/data/seg-lungs-LUNA16/'

class Reader:
    """
    Batch_size is currently unused! Returns all the slices of one subject atm
    """
    def __init__(self, batch_size = 100, shuffle = True, meta_data = 'image_stats.stat', label_path = lbl_path, patch_shape = (64,64)):
        
        if not os.path.isfile(meta_data):
            preprocess.preprocess()
        
        with open('image_stats.stat', 'rb') as read:
            meta_data = pickle.load(read)
    
        self.mean = meta_data['mean']
        self.std = meta_data['std']
        self.n_samples = meta_data['n_samples']
        #self.n_slices  = meta_data['n_slices']
        self.file_names = meta_data['file_names']
        #self.n_batches = int(n_slices/batch_size)
        self.batch_size = batch_size
        self.label_path = label_path
        self.current = 0
        self.shuffle = shuffle
        self.patch_shape = patch_shape

    def __iter__(self):
        return self

    def next(self):
        if self.current >self.n_samples-1:
            raise StopIteration
        else:
            batch, labels = load_itk_images(*self.get_locations())
            batch = batch - self.mean
            labels = labels >= 3

            n_patches = 2

            patch_batch = np.zeros((n_patches*len(batch), 1,) + self.patch_shape, dtype=np.float32)
            patch_labels = np.zeros((n_patches*len(batch), 2), dtype=np.float32)
            
            for i in range(len(batch)):
                image_patches, image_labels = self.patch(batch[i], labels[i], n_patches)
                patch_batch[i*n_patches:i*n_patches+n_patches] = image_patches
                patch_labels[i*n_patches:i*n_patches+n_patches] = image_labels

            
            
            self.current+=1
            return patch_batch, patch_labels


def patch(self, image, labels, n_patches):
    # output:
    # image: (n_patches, 1, 64, 64)
    # label: (n_patches)
    
    

    patches = np.zeros((n_patches, 1, 64, 64), dtype=np.float32)
    patch_labels = np.zeros((n_patches, 2), dtype = np.float32)
    
    n_possible = (image.shape[0]-patch_size+1)*(image.shape[1]-patch_size+1)

    all_labels = np.zeros((n_possible), dtype=np.float32)
    
    #Get all lables in range
    i = 0
    for x in range(image.shape[0]-patch_size+1):
        for y in range(image.shape[0]-patch_size+1):
            all_labels[i] = labels[y+patch_size/2, x+patch_size/2]
            i += 1
            
    #split labels in positive and negative samples (indices)
    neg_labels = np.argwhere(all_labels == 0).flatten()
    pos_labels = np.argwhere(all_labels == 1).flatten()

    
    n_positives = min(len(pos_labels),n_patches/2) #Not always enough positive labels, sometimes even 0.
    n_negatives = n_patches - n_positives #Are always more negative than positive labels
    
    neg_labels2 =  np.random.choice(neg_labels, n_negatives, replace = False)
    pos_labels2 = np.random.choice(pos_labels, n_positives, replace = False)
    
    patch_indices = np.concatenate([neg_labels2, pos_labels2])    
    

    for i, index in enumerate(patch_indices):
        x = index / int(image.shape[0]-patch_size+1) #transform index to coordinates
        y = index % (image.shape[1]-patch_size+1)
        patch_labels[i] = labels[y+patch_size/2, x+patch_size/2] 

        patches[i, 0, :, :] = image[y:y+patch_size, x:x+patch_size]
        

    return patches, patch_labels


def load_itk_images(input_path, target_path):
    itkinput = sitk.ReadImage(input_path)
    numpyinput = sitk.GetArrayFromImage(itkinput)
    
    itktarget = sitk.ReadImage(target_path) 
    numpytarget = sitk.GetArrayFromImage(itktarget)
    #numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    #numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    #resized_input = np.resize(itkinput, (itkinput.shape[0],itkinput.shape[1]/2,itkinput.shape[2]/2))
    #resized_target = np.resize(itktarget, (itktarget.shape[0],itktarget.shape[1]/2,itktarget.shape[2]/2))
    #print resized_input.shape
    #print resized_target.shape
    
    #return itkinput, itktarget
    return numpyinput, numpytarget



if __name__ == "__main__":
    """
   path = "D:/data/subset7/subset7/"
    seg_path = "D:/data/seg-lungs-LUNA16/seg-lungs-LUNA16/"
    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith(".mhd"):
            target,_,_ = load_itk_image(seg_path+filename)
            input_image, _,_ = load_itk_image(path+filename)
            for im_slice in input_image:
                print im_slice.shape
    print i
        """
        
    filename = "D:/data/subset7/subset7/1.3.6.1.4.1.14519.5.2.1.6279.6001.105495028985881418176186711228.mhd"
    labelname = "D:/data/seg-lungs-LUNA16/1.3.6.1.4.1.14519.5.2.1.6279.6001.105495028985881418176186711228.mhd"
    i=0
    for batch, labels in tqdm(Reader()):
       i=1 
        

