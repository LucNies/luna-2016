import getpass
import glob
import os
from itertools import izip

import SimpleITK as sitk
from math import sqrt

import numpy as np
from scipy.ndimage import center_of_mass, label, measurements
from tqdm import tqdm
import matplotlib.pyplot as plt

from prep_annotation import Annotator

if getpass.getuser().lower() == "steven":
    data_dir = os.path.join("F:/Temp/CAD/data")
elif getpass.getuser().lower() == 'the mountain':
    data_dir = os.path.join('D:/data/')

else:
    data_dir = os.path.join("..", "data")

annotation_filename = os.path.join(data_dir, "CSVFILES/annotations.csv")
annotations = []
with open(annotation_filename) as annotationstream:
    annotationstream.next() # Skip header
    for annotation in annotationstream:
        annotations.append(annotation.split(","))


def load_annotation(filename):
    if filename.endswith(".npz"):
        filename = filename[:-4]
    if filename.endswith(".mhd"):
        filename = filename[:-4]
    return [a for a in annotations if a[0] == filename]


def dist(lesion, center):
    z, y, x = float(lesion[1]),  float(lesion[2]),  float(lesion[3])
    x_, y_, z_ = center
    xd = x - x_
    yd = y - y_
    zd = z - z_
    return sqrt(xd*xd + yd*yd + zd*zd)


def calc_conf(vs):
    return np.mean(vs)


def get_centers(predictions):
    min_area = 20
    predictions_b = predictions > 0.5
    lbl, ncenters = label(predictions_b)
    counts = measurements.sum(lbl>0, lbl, index=[i+1 for i in range(ncenters)])
    relevant = [i+1 for i, c in enumerate(counts) if c > min_area]
    #ncenters = len(relevant)
    centers = center_of_mass(predictions_b, lbl, relevant)

    confs = []
    for i in relevant:
        vs = predictions[lbl == (i+1)] # +1 needed since 0 is background
        confs.append(calc_conf(vs))
    return centers, confs


def to_world(center, fn):
    fn = Annotator.search_file(fn)
    itkimage = sitk.ReadImage(fn)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return np.array(center) * np.array(spacing) + np.array(origin)


def one_froc(annotation, predictions, t, filename):
    LL = 0
    LN = 0
    centers, confidence = get_centers(predictions)
    centers = [cc[0] for cc in zip(centers, confidence) if cc[1] >= t]
    im_filename = os.path.basename(filename)[:-4]
    centers = [to_world(center, im_filename) for center in centers]
    for center in tqdm(centers):
        for lesion in annotation:
            if dist(lesion, center) < lesion[4]:
                LL+=1
                LN-=1 # Do not increment LN for we found a lesion
                break # Go to next center
        LN+=1
    return LL, LN


# Not used, result of brainfart
def cumulative(L):
    cur_LL, cur_LN = 0, 0
    cL = []
    for t, LL, LN in L:
        cur_LL += LL
        cur_LN += LN
        cL.append((t, cur_LL, cur_LN))
    return cL


def grab_right_t(L):
    cur_t = L[0][0] # To prevent adding on for loop start
    cur_LL = 0
    cur_LN = 0
    nL = []
    for tup in L:
        if tup[0] != cur_t:
            nL.append((cur_t, cur_LL, cur_LN))
        cur_t, LL, LN = tup
        cur_LL += LL
        cur_LN += LN
    if len(L) > 0:
        nL.append((cur_t, LL, LN)) # Don't forget very last one
    return nL


def count_lesions():
    return len(annotations)


def calculate_froc(filenames, predictions):
    L = []
    for filename, prediction in tqdm(izip(filenames, predictions)):
        annotation = load_annotation(filename)
        thresholds = set(prediction.flatten())
        for t in tqdm(thresholds):
            LL, NL = one_froc(annotation, prediction, t, filename)
            L.append((t, LL, NL))

    L.sort(key=lambda x: x[0])
    L = grab_right_t(L)
    L = [(l[1], l[2]) for l in L]
    nles = float(count_lesions())
    nim = float(len(filenames))
    LF = [(l[0]/nles, l[1]/nim) for l in L]
    return LF


if __name__ == "__main__":

    """
    def generate_annotations(filenames):
        for filename in filenames:
            A = Annotator(filename)
            yield A.get_full(warning=False).astype(bool)

    files = []
    for i in range(1):
        files.extend(glob.glob(os.path.join(data_dir, 'subset%i' % i) + "/*.mhd"))

    files = [x.split(os.sep)[-1] for x in files][:1]


    froc = calculate_froc(files, generate_annotations(files))
    """
    seg_dir = os.path.join("..", "data", "seg_nodules")
    files = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]

    def yield_predictions():
        for f in files:
            yield np.load(f)['arr_0']*1


    froc = calculate_froc(files, yield_predictions())
    plt.scatter(*zip(*froc))
    plt.show()
    raw_input("Press [Enter] to finish")
