from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD


def build_model(n_filters):
	model = Sequential()

	for i in range(4):
		model.add(Convolution3D(2**i * n_filters, 5, 5, 3, init='he_normal', border_mode='same', input_shape=(1, 40, 40, 26)))
		model.add(LeakyReLU())
		model.add(BatchNormalization())
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

	model.add(Flatten())
	model.add(Dropout(p=0.25))
	model.add(Dense(output_dim=512, init='he_normal'))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(p=0.25))
	model.add(Dense(output_dim=2, init='he_normal', activation='softmax'))

	opt = SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model
