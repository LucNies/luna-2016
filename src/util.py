# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 20:49:52 2016

@author: Luc
"""

from tqdm import tqdm
import numpy as np
from scipy.misc import imsave
import os

def save_subject_as_image(image, subject_name, prefix = "", file_path = "D:/data/images/"):
    
    if not os.path.exists(file_path+subject_name):
        os.makedirs(file_path+subject_name)
        
    for i, layer in tqdm(enumerate(image)):
        imsave(file_path + subject_name + "/" + prefix + "slice" + str(i) + ".png", layer)
        
        
