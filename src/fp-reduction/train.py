from iterators import ParallelBatchIterator
import glob
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix

def _search_file(uid):
	for i in range(10):
		path = '/scratch-shared/wijnands/cad/subset' + str(i) + '/' + uid + '.mhd'
		if os.path.isfile(path):
			return sitk.ReadImage(path)

	raise

def _world_to_pixel(world, o, s):
	coordinate = np.array(world).astype(float)
	stretchedVoxelCoord = np.absolute(coordinate - np.array(o))
	voxelCoord = stretchedVoxelCoord / s
	return voxelCoord.astype(int)

def _convert_to_pixel_space(itk, coords):
	origin = itk.GetOrigin()
	spacing = itk.GetSpacing()

	return _world_to_pixel(coords, origin, spacing)

def _generator(rows):
	# Read files and labels
	X = np.zeros((len(rows), 1, 40, 40, 26))
	y = np.zeros((len(rows), 2))

	for i, row in enumerate(rows.iterrows()):
		row = row[1]
		uid = row[0]
		coords = (row[1], row[2], row[3])
		label = np.array([1-row[4], row[4]], dtype=np.int32)

		image_file = _search_file(uid)
		pixel_space_coords = _convert_to_pixel_space(image_file, coords)

		image_array = sitk.GetArrayFromImage(image_file)
		crop = image_array[pixel_space_coords[2]-13:pixel_space_coords[2]+13,
						   pixel_space_coords[1]-20:pixel_space_coords[1]+20,
						   pixel_space_coords[0]-20:pixel_space_coords[0]+20]

		# Transpose to 01t dimension from t01
		crop = crop.transpose(2, 1, 0)

		# Subtract mean and divide by std
		crop = crop + 750.58
		crop = crop / 367.90

		X[i, 0, :crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
		y[i] = label

	return X, y

def _shuffle_and_balance(data):
	# Every epoch, take different negative samples
	epoch = data.iloc[np.random.permutation(len(data))]

	positives = epoch[epoch['class'] == 1]
	negatives = epoch[epoch['class'] == 0]

	# Take only as many negatives as we have positives, randomly
	negatives = negatives.iloc[:len(positives)]

	# Re-merge positives and negatives
	epoch = positives.append(negatives, ignore_index=True)

	# Shuffle again
	epoch = epoch.iloc[np.random.permutation(len(epoch))]

	return epoch


def train():
	# Import here so it does not get forked
	from model import build_model

	model = build_model(n_filters=32)
	model.summary()
	csv = pd.DataFrame.from_csv("/scratch-shared/wijnands/cad/CSVFILES/candidates_V2.csv", index_col=False)
	csv = csv.iloc[np.random.permutation(len(csv))]

	positives = csv[csv['class'] == 1]
	negatives = csv[csv['class'] == 0]

	# Stratified train/val split
	validation_positives = positives.iloc[:int(0.15*len(positives))]
	validation_negatives = negatives.iloc[:int(0.15*len(negatives))]

	train_positives = positives.iloc[int(0.15*len(positives)):]
	train_negatives = negatives.iloc[int(0.15*len(negatives)):]

	validation = validation_positives.append(validation_negatives, ignore_index=True)
	train = train_positives.append(train_negatives, ignore_index=True)

	for epoch in range(10):
		epoch_train = _shuffle_and_balance(train)
		epoch_validation = _shuffle_and_balance(validation)

		training_loss = []
		validation_loss = []
		train_cm = np.zeros((2, 2), dtype=np.int32)
		validation_cm = np.zeros((2, 2), dtype=np.int32)

		# Train
		for X, y in tqdm(ParallelBatchIterator(_generator, epoch_train, batch_size=16, n_producers=4)):
			loss = model.train_on_batch(X, y)
			predictions = model.predict_on_batch(X)

			cm_batch = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1), labels=[0, 1])
			train_cm += cm_batch

			training_loss.append(loss)

		# Validate
		for X, y in tqdm(ParallelBatchIterator(_generator, epoch_validation, batch_size=16, n_producers=4)):
			loss = model.test_on_batch(X, y)
			predictions = model.predict_on_batch(X)

			cm_batch = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1), labels=[0, 1])
			validation_cm += cm_batch

			validation_loss.append(loss)

		train_tn, train_tp, train_fp, train_fn = (train_cm[0][0], train_cm[1][1], train_cm[0][1], train_cm[1][0])
		validation_tn, validation_tp, validation_fp, validation_fn = (validation_cm[0][0], validation_cm[1][1], validation_cm[0][1], validation_cm[1][0])

		print "Epoch #{}: training loss {:.3f}, training accuracy {:.3f}, tp {:d}, tn {:d}, fp {:d}, fn {:d}, recall {:.3f}, precision {:.3f}".format(epoch, np.mean(training_loss), (train_tp + train_tn) / float(train_tp+train_tn+train_fp+train_fn), train_tp, train_tn, train_fp, train_fn, train_tp / float(train_tp + train_fn), train_tp / float(train_tp + train_fp))
		print "Epoch #{}: valid    loss {:.3f}, valid    accuracy {:.3f}, tp {:d}, tn {:d}, fp {:d}, fn {:d}, recall {:.3f}, precision {:.3f}".format(epoch, np.mean(validation_loss), (validation_tp + validation_tn) / float(validation_tp+validation_tn+validation_fp+validation_fn), validation_tp, validation_tn, validation_fp, validation_fn, validation_tp / float(validation_tp + validation_fn), validation_tp / float(validation_tp + validation_fp))

if __name__ == "__main__":
	train()
