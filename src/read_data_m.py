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
import getpass

if getpass.getuser() == 'harmen':
    lbl_path = os.path.join("..", "data", "seg-lungs-LUNA16")
else:
    lbl_path = 'F:/Temp/CAD/data/seg-lungs-LUNA16/'

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

            n_patches = 1

            patch_batch = np.zeros((n_patches*len(batch), 1,) + self.patch_shape, dtype=np.float32)
            patch_labels = np.zeros((n_patches*len(batch), 2), dtype=np.float32)


            for i in range(len(batch)):
                image_patches, image_labels = self.patch(batch[i], labels[i], n_patches)
                patch_batch[i*n_patches:i*n_patches+n_patches] = image_patches
                patch_labels[i*n_patches:i*n_patches+n_patches] = image_labels
            
            
            self.current+=1
            return patch_batch, patch_labels


    def patch(self, image, labels, n_patches=1000):
        # image: (1, 1, 512, 512)
        # label: (1, 1, 512, 512)

        # output:
        # image: (n_patches, 1, 64, 64)
        # label: (n_patches)

        patches = np.zeros((n_patches, 1,) + self.patch_shape, dtype=np.float32)
        patch_labels = np.zeros((n_patches, 2), dtype=np.float32)

        pw, ph = self.patch_shape
        for i in range(n_patches):
            coords = np.random.randint(0, 512 - pw + 1), np.random.randint(0, 512 - ph + 1)
            patches[i, 0, :, :] = image[coords[0]:coords[0]+pw, coords[1]:coords[1]+ph]
            x = coords[0] + pw/2
            y = coords[1] + ph/2
            v = labels[int(x), int(y)]
            patch_labels[i, :] = [1-v, v]

        return patches, patch_labels


    def get_locations(self):
        image_location = self.file_names[self.current]
        split = image_location.split('/')
        label_location = os.path.join(self.label_path, split[-1])
        return image_location, label_location

    def load_itk_images(self, input_path, target_path):
        return load_itk_images(input_path, target_path)


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
        

