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
    sumpt = np.sum(p) + np.sum(t)
    if sumpt == 0:
        return 1.0
    return (np.sum(p[t == 1.0]) * 2.0) / sumpt
	
def recall_img(p,t):
    sumt = np.sum(t)
    if sumt == 0:
        return 1.0
    return np.sum(p[t == 1.0])/ sumt


if __name__ == "__main__":
    print "EVALUATING DICE SCORES\n======================"
    path = "../data/"
    seg_path = "I:/mevisnak/candidatemasks/candidates/"
    pred_path = "I:/mevisnak/predictions/"
    #dices = np.empty(shape = (len(os.listdir(pred_path))))
    recalls = np.zeros(shape = (len(os.listdir(pred_path))/2))
    j = 0
    for i,filename in enumerate(os.listdir(pred_path)):
        if filename.endswith(".mhd"):
            #target = load_itk_image(seg_path+filename)
            target = np.load(seg_path+filename+".npz")
            target = target['arr_0']*1.0
            #print np.sum(target)
            prediction = load_itk_image(pred_path+filename)
            #print "Original images:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            target = stats.threshold(target, None, 0, 1)
            target = stats.threshold(target, 0, None, 0)
            target *=1.0
            #print np.sum(target)
            prediction = stats.threshold(prediction, None, 0, 1)
            prediction = stats.threshold(prediction, 0, None, 0)
            #print prediction
            #print "After thresholding:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            prediction *=1.0            
            recalls[j] = recall_img(prediction, target)
            print 'pred {0}, tar {1}, rec{2}'.format(np.sum(prediction), np.sum(target), recalls[j])
            #dices[j] = dice_score_img(prediction,target)
            #print np.sum(prediction*target)
            j += 1
            print '{0}\r'.format(j),
    print recalls
    #print dices
    print
    #mean = np.mean(dices)
    #std = np.std(dices)
    rmean = np.mean(recalls)
    rstd = np.std(recalls)
    #print "Dice score mean: {0}, std: {1}".format(mean, std) 
    print "Recall mean: {0}, std: {1}".format(rmean, rstd) + "\n======================"
