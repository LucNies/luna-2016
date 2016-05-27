import numpy as np
from scipy import sparse
import os
import SimpleITK as sitk
import getpass
import glob
from tqdm import tqdm

from read_data_m import load_itk_images

if getpass.getuser().lower() == "steven":
    data_dir = os.path.join("F:/Temp/CAD/data")
elif getpass.getuser().lower() == 'the mountain':
    data_dir = os.path.join('D:/data/')
    
else:
    data_dir = os.path.join("..", "data")

annotation_filename = os.path.join(data_dir, "CSVFILES/annotations.csv")
assert os.path.exists(annotation_filename), "Please put annotations.csv in {}".format(
    os.path.abspath(annotation_filename))


class Annotator:
    """Generates annotations for nodules. One Annotator per .mhd file is used. Use get or get_full to obtain slices or
    the full matrix respectively. Do not forget to give non-default slice shapes."""

    def __init__(self, filename, annotation_filename = annotation_filename):
        """
        Initializes the annotation by building a sparse 3D matrix of the relevant file using the annotation file.
        Please make sure that the annotation filename is correct.
        Args:
            filename: The .mhd-filename of the current patient
        """
        self.filename = filename
        self.annotation_filename = annotation_filename
        self.annotation = self.prep()

    @staticmethod
    def dist(point, ellipse):
        return np.sum(p*p / (float(e*e)+0.00001) for p,e in zip(point, ellipse))

    def generate_coos(self, c, d):
        r = tuple(di / 2 for di in d)
        x, y, z = c
        for i in range(-r[0]-5, r[0] + 5):
            for j in range(-r[1]-5, r[1] + 5):
                for k in range(-r[2]-5, r[2] + 5):
                    if Annotator.dist((i, j, k), r) <= 1:# was 2 once
                        if x+i >= 0 and y+j >= 0 and z+k >= 0:
                            if x+i < self.slice_shape[0] and y+j < self.slice_shape[1] and z+k < self.slice_shape[2]:
                                yield (x + i, y + j, z + k)

    @staticmethod
    def world_to_pixel(world, o, s):
        coordinate = np.array(world).astype(float)[:3]
        d = float(world[3])
        stretchedVoxelCoord = np.absolute(coordinate - np.array(o))
        voxelCoord = stretchedVoxelCoord / s
        voxelDist = np.array([d, d, d]) / s
        return voxelCoord.astype(int), voxelDist.astype(int)

    @staticmethod
    def search_file(fn):
        if os.path.exists(fn):
            return fn
        for i in range(10):
            fullname = os.path.join(data_dir, "subset{}".format(i), fn)
            if os.path.exists(fullname):
                return fullname
        raise Exception("File not found: {}".format(fn))

    def get_spacing(self, fn):
        fn = self.search_file(fn)
        itkimage = sitk.ReadImage(os.path.join(data_dir, fn))
        origin = np.array(list(reversed(itkimage.GetOrigin())))
        spacing = np.array(list(reversed(itkimage.GetSpacing())))
        self.slice_shape = np.array(list(reversed(itkimage.GetSize())))
        return origin, spacing

    def prep(self):
        orig, spacing = self.get_spacing(self.filename)
        coos = []
        with open(self.annotation_filename) as annotations:
            for annotation in annotations:
                filename_ , z, y, x, d = annotation.split(",") # z,y,x is the coordinate encoding for some reason
                if filename_ + ".mhd" == self.filename:
                    c_, d_ = self.world_to_pixel((x, y, z, d), orig, spacing)
                    for coo in self.generate_coos(c_, d_):
                        coos.append(coo)

        mat3 = dict()
        for slice in set(c[0] for c in coos):
            cur_coos = [c[1:] for c in coos if c[0] == slice]
            mat = sparse.coo_matrix(([1] * len(cur_coos), zip(*cur_coos)), shape=self.slice_shape[1:])
            mat3[slice] = mat
        return mat3

    def get(self, slice, dense=False):
        """
        Obtains one slice of annotations.
        Args:
            slice: Slice index.
            dense: Return np.array or sp.coo_matrix?

        Returns: One slice shaped as given in intialization

        """
        if slice not in self.annotation:
            mat = sparse.coo_matrix(self.slice_shape[1:])
        else:
            mat = self.annotation[slice]
        if dense:
            return mat.todense()
        else:
            return mat

    def get_full(self, warning=True):
        """
        Generates the full 3D annotation in dense for (since 3D coo-matrices do not exist).
        Args:
            warning: True to warn at run-time

        Returns: Full 3D dense np.array with all annotations

        """
        if warning:
            print "Creating a full dense {}-matrix. This might be too big for your RAM".format(self.slice_shape)
        mat = np.zeros(self.slice_shape)
        for slice in self.annotation.iterkeys():
            mat[slice, :, :] = self.get(slice, True)
        return mat


if __name__ == "__main__":

    # Get files
    files = []
    for i in range(10):
        files.extend(glob.glob(os.path.join(data_dir, 'subset%i' % i) + "/*.mhd"))
    
    files = [x.split(os.sep)[-1] for x in files]

    for f in tqdm(files):
        A = Annotator(f)
        m = A.get_full(warning=False).astype(bool)
        """
        import matplotlib.pyplot as plt
        img, _ = load_itk_images(os.path.join(data_dir, "subset0", f), os.path.join(data_dir, "subset0", f))
        for i, im in zip(m, img):
            if True in i:
                plt.figure(1)
                plt.imshow(i)
                plt.figure(2)
                plt.imshow(im)
                plt.show()
        """
        np.savez_compressed(os.path.join(data_dir, 'candidates', f), m)

        # Load: np.load(os.path.join(data_dir, 'candidates', f + '.npz'))['arr_0']
