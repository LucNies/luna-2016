# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:39:52 2016

@author: Luc-squad
"""

import SimpleITK as sitk
import numpy as np
import csv
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import util


class Reader:
    
    def __init__(self, file_path = "D:/data/subset9/"):
        self.file_path = file_path

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
    image, label = load_itk_images(filename, labelname)
    label = label == 3
    util.save_subject_as_image(label, "105495028985881418176186711228", prefix = "label")
    print "done"

