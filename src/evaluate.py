# -*- coding: utf-8 -*-
"""
Created on Thu May 5 17:39:52 2016

@author: Inez
"""
from __future__ import division
#from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import scipy.stats as stats
import os

def load_itk_image(input_path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(input_path))

    return image

def dice_score_img(p,t):
    return np.sum(p[t == 1]) * 2.0 / (np.sum(p) + np.sum(t))

if __name__ == "__main__":
    print "EVALUATING DICE SCORES\n======================"
    path = "../data/"
    seg_path = "../data/seg-lungs-LUNA16/"
    pred_path = "../data/pred/"
    dices = np.empty(shape = (1))
    j = 0
    for i,filename in enumerate(os.listdir(path)):
        if filename.endswith(".mhd"):
            target = load_itk_image(seg_path+filename)
            prediction = load_itk_image(pred_path+filename)
            #print "Original images:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            target = stats.threshold(target, None, 0, 1)
            target = stats.threshold(target, 0, None, 0)
            prediction = stats.threshold(prediction, None, 0, 1)
            prediction = stats.threshold(prediction, 0, None, 0)
            #print "After thresholding:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            dices[j] = dice_score_img(prediction,target)
            j += 1

    mean = np.mean(dices)
    std = np.std(dices)
    print "Dice score mean {0}, std: {1}".format(mean, std) + "\n======================"
