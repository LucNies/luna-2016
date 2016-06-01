import os
import numpy as np
from scipy.ndimage import measurements
from tqdm import tqdm


def filter_counts(counts):
    return np.argsort(counts)[-2:] # Top 2


def connect(img):
    mask, n = measurements.label(img)
    counts = measurements.sum(img, mask)
    lung_ids = filter_counts(counts)  # TODO: Pak alleen de longen
    def f(v): return v in lung_ids
    return f(mask)

def rename(fn):
    if "".endswith(".mhd.npz"):
        return "{}{}{}".format(fn[:-8], "_connected", fn[-8:])
    else:
        raise Exception("File name too complicated for me")


if __name__ == "__main__":
    seg_dir = os.path.join("..", "segmentations", "nodules")
    files = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]

    def yield_predictions():
        for f in files:
            print "Processing {}".format(f)
            yield f, np.load(f, mmap_mode='r')['arr_0']*1

    for fn, prediction in tqdm(yield_predictions()):
        np.savez_compressed(rename(fn), connect(prediction))
