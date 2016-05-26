# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 20:49:52 2016

@author: Luc
"""

from tqdm import tqdm
import numpy as np
from scipy.misc import imsave
import os
import matplotlib.pyplot as plt

def save_subject_as_image(image, subject_name, prefix = "", file_path = "D:/data/images/"):
    
    if not os.path.exists(file_path+subject_name):
        os.makedirs(file_path+subject_name)
        
    for i, layer in tqdm(enumerate(image)):
        imsave(file_path + subject_name + "/" + prefix + "slice" + str(i) + ".png", layer)
        
        
"""
Left plot is segmentation, left is the prediction
"""        
def view_nodules(file_name):
    pred_path = "../segmentations/nodules/"
    seg_path = "D:/data/candidates/"
    segmentation = np.load(seg_path+file_name)['arr_0']*1
    prediction = np.load(pred_path+file_name)['arr_0']*1
    

    for i in tqdm(range(len(segmentation))):
        if segmentation[i].sum() > 0:            
            f, axarr = plt.subplots(2, sharex=True)
            #plt.imshow(prediction[i, :, :], cmap='gray')
            axarr[0].imshow(segmentation[60, : , :], cmap='gray')
            axarr[0].set_title('Segmentation mask, slice {}'.format(i))
            axarr[1].imshow(prediction[60, : , :], cmap='gray')
            axarr[1].set_title('predictions, slice {}'.format(i))
            plt.show()
    
def dice_score(conf_matrix):
    tp = conf_matrix[0][0]
    fn = conf_matrix[0][1]
    fp = conf_matrix[1][0]
    
    return tp*2. / (tp+fp+tp+fn)
    

if __name__ == '__main__':
    view_nodules('1.3.6.1.4.1.14519.5.2.1.6279.6001.114914167428485563471327801935.mhd.npz')