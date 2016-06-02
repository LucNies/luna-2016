from __future__ import division
import os
import numpy as np
from scipy.ndimage import measurements
import matplotlib.pyplot as plt 
import evaluate
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def filter_counts(counts):
    return np.argsort(counts)[-1] # Top 2


def connect(img):
    mask, n = measurements.label(img)
    print "TQDM"
    counts = measurements.sum(img, mask, range(n))
    print "TQDM.. more"
    print "{}{}{}".format(len(counts), "/", n)
    lung_ids = filter_counts(counts)  # TODO: Pak alleen de longen
    print "TQDM thrice"
    
    return (mask == lung_ids).astype(int)
    
def rename(fn):
    if fn.endswith(".mhd.npz"):
        return "{}{}{}".format(fn[:-8], "_connected", fn[-8:])
    else:
        raise Exception("File name too complicated for me")


def view(ndimg):
    #s = ndimg[100]
    for x in range(50,200,10):
        plot = plt.imshow(ndimg[x])
        plot.set_cmap('gray')
        plt.show()


if __name__ == "__main__":
    seg_dir = os.path.join("..", "segmentations", "nodules")

    files = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]
    def yield_predictions():
        for f in files:
            print "Processing {}".format(f)
            yield f, np.load(f, mmap_mode='r')['arr_0']*1

    for fn, prediction in tqdm(yield_predictions()):
        np.savez_compressed(rename(fn), connect(prediction))
    """
    filename = files[125]
    seg_path = "I:/mevisnak/truth/"
    truth = evaluate.load_itk_image(seg_path+os.path.basename(filename)[:-4])
    
    seg = np.load(filename)['arr_0']*1
    connected = connect(seg)
    print set(truth.flatten())
    print truth.mean()
    print connected.mean()
    print (truth == connected).sum()
    print evaluate.dice_score_img(connected, truth >= 1)   
    print evaluate.dice_score_img(seg, truth >= 1)    
    
    #view(connected)
    #view(seg)
    """
  
   