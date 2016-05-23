# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:39:52 2016

@author: Luc
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
import time


if getpass.getuser() == 'harmen':
    lbl_path = os.path.join("..", "data", "seg-lungs-LUNA16")
else:
    lbl_path = 'D:/data/candidates/'

patch_size = 64


class NoduleReader:

    """
    Class balance = n_nodule_pixels/n_non_nodule_pixels
    """
    def __init__(self, batch_size = 1000, shuffle = True, class_balance = 0.8, meta_data = 'image_stats.stat', label_path = lbl_path, patch_shape = (64,64)):
        
        if not os.path.isfile(meta_data):
            preprocess.preprocess()
        
        with open(meta_data, 'rb') as read:
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
        self.current_slice = 0
        self.shuffle = shuffle
        self.patch_shape = patch_shape
        self.class_balance = class_balance

    def __iter__(self):
        return self


    """
    Reads through the slices of a subject until one with a candidate is detected and extracts patches from that slice, with classes balaced 
    according to self.class_balance. Reapats this until n = batch_size patches are extracted
    """
    def next(self):
        if self.current > self.n_samples-1:
            raise StopIteration
        else:

            n_patches = 100
            n_batches = 0;
            patch_batch = np.zeros((self.batch_size, 1,) + self.patch_shape, dtype=np.float32)
            patch_labels = np.zeros((self.batch_size, 2), dtype=np.float32)
            
            #while the batch size is not full and while there are subjects left
            while n_batches < self.batch_size and self.current < self.n_samples:

                batch, labels = load_subject(*self.get_locations())
                batch = batch - self.mean
                 
                #loop over the slices starting from the current slice (0 if new subject, higher if batch was full before while there were still some slices left)
                for i in np.arange(self.current_slice, len(labels)):
                    #print i, label.sum()
                    label = labels[i]
                    
                    if label.sum() > 0: # so there is a nodule in the slice
                        #print n_batches, n_batches*n_patches+n_patches, self.batch_size
                        image_patches, image_labels = self.patch(batch[i], label, n_patches) #the actual patching
                        patch_batch[n_batches*n_patches:n_batches*n_patches+n_patches] = image_patches
                        patch_labels[n_batches*n_patches:n_batches*n_patches+n_patches] = image_labels
                        n_batches += 1
                        self.current_slice += 1 
                        
                        #batch is completed but still in subject still has some slices left
                        if n_batches*n_patches >= self.batch_size:
                            
                            #Shuffle batches
                            indices = np.arange(self.batch_size)
                            np.random.shuffle(indices)                            
                            
                            return patch_batch[indices], patch_labels[indices]
                            
                self.current_slice = 0
                self.current+=1
                
                
                
                    
                       
            
            
            #If n_batch is not reached yet, but there are no more subjects
            if self.current > self.n_samples-1:
                patch_batch = patch_batch[:n_batches*n_patches]
                patch_labels = patch_labels[:n_batches*n_patches]

            #Shuffle batches
            indices = np.arange(n_batches)
            np.random.shuffle(indices)

            return patch_batch[indices], patch_labels[indices]


    def patch(self, image, labels, n_patches):
        # output:
        # image: (n_patches, 1, 64, 64)
        # label: (n_patches)
              
    
        patches = np.zeros((n_patches, 1, 64, 64), dtype=np.float32)
        patch_labels = np.zeros((n_patches, 2), dtype = np.float32)
        
        n_possible = (image.shape[0]-patch_size+1)*(image.shape[1]-patch_size+1)
    
        all_labels = np.zeros((n_possible), dtype=np.float32)
        
        #Get all lables in range
        all_labels = labels[patch_size/2 :  image.shape[0]- patch_size/2 + 1, patch_size/2 : image.shape[1] - patch_size/2 + 1].flatten()
     
        #split labels in positive and negative samples (indices)
        neg_labels = np.argwhere(all_labels == 0).flatten()
        pos_labels = np.argwhere(all_labels == 1).flatten()
        
        
        n_positives = min(len(pos_labels),n_patches*self.class_balance) #Not always enough positive labels, sometimes even 0.
        n_negatives = n_patches - n_positives #Are always more negative than positive labels
        
        #randomly pick labels
        neg_labels =  np.random.choice(neg_labels, n_negatives, replace = False)
        pos_labels = np.random.choice(pos_labels, n_positives, replace = False)
        
        patch_indices = np.concatenate([neg_labels, pos_labels])    
        
        
        for i, index in enumerate(patch_indices):
            x = index / (image.shape[0]-patch_size+1 )#transform index to coordinates
            y = index % (image.shape[1]-patch_size+1)
            patch_labels[i] = [1 - labels[x + patch_size/2, y + patch_size/2], labels[x + patch_size/2, y + patch_size/2]]
    
            patches[i, 0, :, :] = image[x:x+patch_size, y:y+patch_size]
            
        patches = self.augment_patches(patches) #Augment all patches (both positive and negative samples, not really needed, but a lot of work to split this)


        return patches, patch_labels

    def get_locations(self):
        image_location = self.file_names[self.current]
        split = image_location.split('/')
        label_location = os.path.join(self.label_path, (str(split[-1]) + '.npz')) 

        
        return image_location, label_location
    
    def augment_patches(self, patches):
        
        augment_option = np.random.randint(0, 4, len(patches))
        for i,patch in enumerate(patches):
            if augment_option[i] == 1:#Transpose
                patches[i][0] = patch[0].transpose()
            elif augment_option[i] == 2: #flip left-right
                patches[i] = np.fliplr(patch)
            elif augment_option[i] == 3:#flip upside down
                patches[i] = np.flipud(patch)
                
            #0 is default, does nothing
        
        return patches
            
        
    

def load_subject(input_path, target_path):
    itkinput = sitk.ReadImage(input_path)
    numpyinput = sitk.GetArrayFromImage(itkinput)
    

    candidates = np.load(target_path)['arr_0']


    return numpyinput, candidates



if __name__ == "__main__":
        
    reader = NoduleReader()
    for patch, labels in tqdm(reader):
        print labels.sum()

