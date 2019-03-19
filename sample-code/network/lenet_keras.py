# Author: Taylor Guo, taylorguo@126.com

'''
Keras                  2.1.0
Keras-Applications     1.0.7
Keras-Preprocessing    1.0.8
tensorboard            1.12.2
tensorflow             1.12.0
tensorflow-tensorboard 0.4.0
numpy                  1.14.5
opencv-python          3.4.1.15

paper: (http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
'''

# LeNet-Keras for mnist handwriting digital image classification

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
import cv2, os,datetime

class LeNet:
	@staticmethod
	def build(channels, height, width, classes, activation="relu", weights_path=None):

		input_shape = (height, width, channels)

		model = Sequential()

		model.add(Conv2D(16,(5,5),activation=activation, padding="same", input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Conv2D(32, (5,5), activation=activation, padding="same"))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Flatten())
		model.add(Dense(128, activation=activation))
		model.add(Dense(classes, activation="softmax"))

		if weights_path is not None:
			model.load_weights(weights_path)

		model.summary()

		return model

	@staticmethod
	def load_dataset_mnist():
		print("\t Downloading MNIST dataset ...")

		(training_data, training_labels),(test_data, test_labels) = mnist.load_data()

		training_data = training_data.reshape((60000, 28,28,1))
		test_data = test_data.reshape((10000, 28,28,1))

		training_data = training_data.astype("float32") / 255.0
		test_data = test_data.astype("float32") / 255.0

		training_labels = to_categorical(training_labels)
		test_labels = to_categorical(test_labels)

		return ((training_data, training_labels),(test_data, test_labels))

	@staticmethod
	def train(weight_path=None, load_weights=False, save_weights=True):

		model = LeNet.build(channels=1, height=28, width=28, classes=10)
		model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])

		(train_d, train_l),(test_d, test_l) = LeNet.load_dataset_mnist()

		if load_weights==False:
			print("\t Start training ...")
			model.fit(train_d, train_l, batch_size=128, epochs=20, verbose=1)

			print("\t Now evaluating ...")
			(loss, accuracy) = model.evaluate(test_d, test_l, batch_size=128, verbose=1)

			print("\t Accuracy: {:.2f}%".format(accuracy*100))
		else:
			pass
			# load_weights from weight_path

		if save_weights==True:
			print("\t Save trained weights to file ...")
			if not os.path.exists("models"):
				os.mkdir("models")
			weight_file = "LeNet_{:%Y%m%dT%H%M%S}.h5".format(datetime.datetime.now())
			model.save_weights(os.path.join("models", weight_file), overwrite=True)

	@staticmethod
	def inference():
		pass


if __name__ == '__main__':

    LeNet.train(save_weights=False)