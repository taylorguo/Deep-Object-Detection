# Author: Taylor Guo, taylorguo@126.com
# Python 3.6.7
'''
Keras                  2.1.0
Keras-Applications     1.0.7
Keras-Preprocessing    1.0.8
tensorboard            1.12.2
tensorflow             1.12.0
tensorflow-tensorboard 0.4.0
tflearn                0.3.2
numpy                  1.14.5
opencv-python          3.4.1.15

paper: (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
PPT: (http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
'''

# AlexNet-Keras for oxflower17 image classification

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from tflearn.datasets import oxflower17
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
import cv2, os,datetime

class AlexNet:

	@staticmethod
	def build(channels, height, width, classes, activation="relu", weights_path=None):

		input_shape = (height, width, channels)

		model = Sequential()

		model.add(Conv2D(96,(11,11), strides=(4,4), input_shape=input_shape))
		model.add(BatchNormalization())
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Conv2D(256, (5,5), strides=(2,2)))
		model.add(BatchNormalization())
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Conv2D(384,(3,3), strides=(2,2), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation(activation))

		model.add(Conv2D(384, (3, 3), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation(activation))

		model.add(Conv2D(256, (3, 3), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Flatten())

		model.add(Dense(4096, activation=activation))
		model.add(Dropout(0.5))

		model.add(Dense(4096, activation=activation))
		model.add(Dropout(0.5))

		model.add(Dense(1000, activation=activation))
		model.add(Dropout(0.5))

		model.add(Dense(classes, activation="softmax"))

		if weights_path is not None:
			model.load_weights(weights_path)

		model.summary()

		return model

	@staticmethod
	def load_dataset_oxflower17():
		print("\t Downloading Oxford Flower17 dataset ...")

		training_data, training_labels = oxflower17.load_data(one_hot=True)

		return (training_data, training_labels)

	@staticmethod
	def train(weight_path=None, load_weights=False, save_weights=True):

		model = AlexNet.build(channels=3, height=224, width=224, classes=17)
		model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])

		(train_d, train_l) = AlexNet.load_dataset_oxflower17()

		if load_weights==False:
			print("\t Start training ...")
			model.fit(train_d, train_l, batch_size=128, epochs=100, verbose=1, validation_split=0.3, shuffle=True)
		else:
			pass
			# load_weights from weight_path

		if save_weights==True:
			print("\t Save trained weights to file ...")
			if not os.path.exists("models"):
				os.mkdir("models")
			weight_file = "AlexNet_{:%Y%m%dT%H%M%S}.h5".format(datetime.datetime.now())
			model.save_weights(os.path.join("models", weight_file), overwrite=True)

if __name__ == '__main__':
    AlexNet.train(save_weights=False)