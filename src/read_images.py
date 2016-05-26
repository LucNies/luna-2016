# -*- coding: utf-8 -*-
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
    lbl_path = 'D:/data/seg-lungs-LUNA16/'


class ImageReader:

    def __init__(self, meta_data = 'test_set.stat', label_path = lbl_path):
        
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
        self.label_path = label_path
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        if self.current >10:#self.n_samples-1:
            raise StopIteration
        else:

            image_location, label_locations, subject_name = self.get_locations()
            subject, segmentation = load_itk_images(image_location, label_locations)
            subject = subject - self.mean
            segmentation = segmentation >= 3
            print self.get_locations()
            subject = subject.astype(np.float32)
            segmentation = segmentation.astype(np.float32)
            
            self.current+=1
            return subject, segmentation, subject_name


    def get_locations(self):
        
        image_location = self.file_names[self.current]
        split = image_location.split('/')
        subject_name = split[-1]
        #label_location = os.path.join(self.label_path, split[-1]) does not work on windows 
        label_location = self.label_path + subject_name
        
        return image_location, label_location, subject_name
    
    

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
    for batch, labels in tqdm(ImageReader()):
       i=1 
        

