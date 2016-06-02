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
import connect_lungs
from sklearn.metrics import confusion_matrix

def load_itk_image(input_path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(input_path))

    return image

def dice_score_img(p,t):
    sumpt = np.sum(p) + np.sum(t)
    if sumpt == 0:
        return 1.0
    return (np.sum(p[t == 1]) * 2.0) / sumpt

def precision_img(p,t):
     conf = confusion_matrix(t, p, labels = [0,1])
     tn = conf[0][0]
     fp = conf[0][1]
     fn = conf[1][0]
     tp = conf[1][1]
     
     return tp/(tp+fp)

    
	
def recall_img(p,t):
    sumt = np.sum(t)
    if sumt == 0:
        return 1.0
    return np.sum(p[t == 1.0])/ sumt
    
def evaluate_lungs():
    print "EVALUATING DICE SCORES\n======================"
    path = "../data/"
    seg_path = "I:/mevisnak/truth/"
    pred_path = "I:/mevisnak/fcn_segs/lung_masks/"
    dices = np.empty(shape = (len(os.listdir(pred_path))))
    #recalls = np.zeros(shape = (len(os.listdir(pred_path))))
    j = 0
    for i,filename in enumerate(os.listdir(pred_path)):
        if filename.endswith(".npz"):
            #target = load_itk_image(seg_path+filename)
            target = np.load(pred_path+filename)
            target = connect_lungs.connect(target['arr_0']*1.0)
            #print np.sum(target)
            #prediction = np.load(pred_path+filename)
            prediction = load_itk_image(seg_path+filename[:-4])
            #print "Original images:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            target = stats.threshold(target, None, 0, 1)
            target = stats.threshold(target, 0, None, 0)
            #print np.sum(target)
            prediction = stats.threshold(prediction, None, 0, 1)
            prediction = stats.threshold(prediction, 0, None, 0)
            #print prediction
            #print "After thresholding:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            prediction *=1.0            
            #recalls[j] = recall_img(prediction, target)
            #print 'pred {0}, tar {1}, rec{2}'.format(np.sum(prediction), np.sum(target), recalls[j])
            dices[j] = dice_score_img(prediction,target)
            #print np.sum(prediction*target)
            j += 1
            print '{0}\r'.format(j),
    #print recalls
    #print dices
    print
    mean = np.mean(dices)
    std = np.std(dices)
    #rmean = np.mean(recalls)
    #rstd = np.std(recalls)
    print "Dice score mean: {0}, std: {1}".format(mean, std) 
    #print "Recall mean: {0}, std: {1}".format(rmean, rstd) + "\n======================"

def evaluate_nodules(): 
    print "EVALUATING DICE SCORES\n======================"
    pred_path = "../segmentations/nodules/"
    seg_path = "D:/data/candidates/"
    path = "../data/"

    #dices = np.empty(shape = (len(os.listdir(pred_path))))
    recalls = np.array([])
    precision = np.array([])
    j = 0
    for i,filename in enumerate(os.listdir(pred_path)):
        #target = load_itk_image(seg_path+filename)
        target = np.load(seg_path+filename)
        target = target['arr_0']*1.0
        if target.sum() > 0:
        #print np.sum(target)
            prediction = np.load(pred_path+filename)
            prediction = prediction['arr_0']*1.0
            #print "Original images:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            #target = stats.threshold(target, None, 0, 1)
            #target = stats.threshold(target, 0, None, 0)
            #target *=1.0
            #print np.sum(target)
            #prediction = stats.threshold(prediction, None, 0, 1)
            #prediction = stats.threshold(prediction, 0, None, 0)
            #print prediction
            #print "After thresholding:\n  TARGET max: {0}, min: {1}, mean: {2}\n  PREDICTION: max: {3}, min: {4}, mean: {5}".format(np.max(target), np.min(target), np.mean(target), np.max(prediction), np.min(prediction), np.mean(prediction))
            
            #precision = np.append(precision, precision_img(prediction.flatten(), target.flatten()))
            recalls = np.append(recalls, recall_img(prediction, target))
            precision = np.append(precision, precision_img(prediction.flatten(), target.flatten()))
            print 'pred {}, tar {}, rec{}, prec {}'.format(np.sum(prediction), np.sum(target), recalls[j], precision[j])
           
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
    print "Recall mean: {}, std: {}".format(rmean, rstd) + "\n======================"
    print "Precision mean: {}, std: {}".format(precision.mean(), precision.std()) + "\n======================"



if __name__ == "__main__":
    evaluate_lungs()
