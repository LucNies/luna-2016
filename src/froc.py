import getpass
import glob
import os
from itertools import izip

import SimpleITK as sitk
from math import sqrt

import numpy as np
from scipy.ndimage import center_of_mass, label, measurements, find_objects
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
    for _annotation in annotationstream:
        annotations.append(_annotation.split(","))

center_sizes = dict()

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


def adjust_slice_coordinates(cor, slices):
    rel_cor = tuple([s.start for s in slices])  # Top-left
    return tuple(c + rc for c, rc in zip(cor, rel_cor))


def get_centers(predictions):
    min_area = 20
    predictions_b = (predictions > 0.5)
    lbl, ncenters = label(predictions_b)
    slices = find_objects(lbl)
    centers = []
    confs = []
    for i, slice in enumerate(slices):
        prediction_b = predictions_b[slice]
        cur_label = lbl[slice] == (i+1)
        count = measurements.sum(prediction_b, cur_label)
        if center_sizes[count] is not None:
            center_sizes[count]+=1
        else:
            center_sizes[count] =1
        if count > min_area:
            centers.append(adjust_slice_coordinates(center_of_mass(cur_label, prediction_b), slice))
            vs = predictions[slice][cur_label]
            confs.append(calc_conf(vs))

    return centers, confs


def to_world(center, fn):
    fn = Annotator.search_file(fn)
    itkimage = sitk.ReadImage(fn)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return np.array(center) * np.array(spacing) + np.array(origin)


def one_froc(centers, confidence, annotation, t, filename):
    LL = 0
    LN = 0
    centers = [cc[0] for cc in zip(centers, confidence) if cc[1] >= t]

    im_filename = os.path.basename(filename)[:-4]
    wrld_centers = []
    for center in centers:
        wrld_centers.append(to_world(center, im_filename))
    for center in tqdm(wrld_centers):
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
    cur_t = L[0][0] # To prevent nL.append on first iteration
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
    files_tqdm = tqdm(izip(filenames, predictions), total=len(filenames))
    for filename, prediction in files_tqdm:
        try:
            Annotator.search_file(os.path.basename(filename)[:-4]) # Check if file exists
            annotation = load_annotation(filename)
            thresholds = set(prediction.flatten())
            if len(thresholds) > 100:
                thresholds = sorted(thresholds)
                thresholds = thresholds[::len(thresholds)/100]

            print "Calculating FROC for {}".format(filename)
            centers, confidence = get_centers(prediction)
            for t in thresholds:
                LL, LN = one_froc(centers, confidence, annotation, t, filename)
                L.append((t, LL, LN))
            print "Processed {}".format(filename)
        except Exception:
            print "Skipping {}".format(filename)
    files_tqdm.close()

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
    #files = files[18:]

    def yield_predictions():
        for f in files:
            print "Processing {}".format(f)
            yield np.load(f, mmap_mode='r')['arr_0']*1

    froc = calculate_froc(files, yield_predictions())
    np.save("../data/froc", froc)
    plt.scatter(*zip(*froc))
    plt.show()
    c_sizes = sorted(center_sizes.keys())
    plt.scatter(c_sizes, [center_sizes[k] for k in c_sizes])
    plt.show()
    raw_input("Press [Enter] to finish")

