import numpy as np
import matplotlib.pyplot as plt

def visualize(file_path):
	f = np.load(file_path)
	r = f['arr_0']

	for i in r:
		plt.imshow(i)
		plt.show()

if __name__ == "__main__":
	# Well segmented:
	visualize("lung_masks/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd.npz")

	# Poorly segmented:
	visualize("lung_masks/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd.npz")
