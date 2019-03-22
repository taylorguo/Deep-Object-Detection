# Author: Taylor Guo, taylorguo@126.com
# Python 3.6.7
'''
Keras                  2.1.0
Keras-Applications     1.0.7
Keras-Preprocessing    1.0.8
tensorboard            1.12.2
tensorflow             1.12.0
tensorflow-tensorboard 0.4.0
numpy                  1.14.5
opencv-python          3.4.1.15

paper: (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
PPT: (http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
'''

from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras.applications.vgg19 import VGG19

from keras.datasets import cifar100

from keras.utils import to_categorical
from keras.optimizers import SGD, Adam

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import numpy as np
import os,datetime

class VGG19_Net:

	@staticmethod
	def build(input_shape, activation="relu", padding="same", classes=100):

		input = Input(shape=input_shape, name="model_input")

		# Block 1
		x = Conv2D(64, (3, 3), padding=padding, activation=activation, name="block1_conv1")(input)
		x = Conv2D(64, (3, 3), padding=padding, activation=activation, name="block1_conv2")(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

		# Block 2
		x = Conv2D(128, (3, 3), padding=padding, activation=activation, name="block2_conv1")(x)
		x = Conv2D(128, (3, 3), padding=padding, activation=activation, name="block2_conv2")(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

		# Block 3
		x = Conv2D(256, (3, 3), padding=padding, activation=activation, name="block3_conv1")(x)
		x = Conv2D(256, (3, 3), padding=padding, activation=activation, name="block3_conv2")(x)
		x = Conv2D(256, (3, 3), padding=padding, activation=activation, name="block3_conv3")(x)
		x = Conv2D(256, (3, 3), padding=padding, activation=activation, name="block3_conv4")(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")

		# Block 4
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block4_conv1")(x)
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block4_conv2")(x)
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block4_conv3")(x)
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block4_conv4")(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")

		# Block 5
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block5_conv1")(x)
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block5_conv2")(x)
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block5_conv3")(x)
		x = Conv2D(512, (3, 3), padding=padding, activation=activation, name="block5_conv4")(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")

		# classification block
		x = Flatten(name="flatten")(x)
		x = Dense(512, activation=activation, name="fc1")(x)
		x = Dense(128, activation=activation, name="fc2")(x)
		x = Dense(classes, activation="softmax", name="prediction")(x)

		model = Model(input, x, name="VGG19_Net")

		model.summary()

		return model


	@staticmethod
	def build_from_config(input_shape, activation="relu", padding="same", classes=100):

		model_config = {"block1": {"layers": 2, "filter": 64},
		                "block2": {"layers": 2, "filter": 128},
		                "block3": {"layers": 4, "filter": 256},
		                "block4": {"layers": 4, "filter": 512},
		                "block5": {"layers": 4, "filter": 512}
		                }

		input = Input(shape=input_shape, name="model_input")

		x = input
		for block in model_config.keys():
			for layer_nb in range(model_config[block]["layers"]):
				kernel = model_config[block]["filter"]
				layer_name = "%s_conv%d"%(block, layer_nb+1)
				# print(layer_name)
				x = Conv2D(kernel, (3, 3), activation=activation, padding=padding, name=layer_name)(x)
			x = MaxPooling2D((2, 2), strides=(2, 2), name="%s_pool"%block)(x)

		# classification block
		x = Flatten(name="flatten")(x)
		x = Dense(512, activation=activation, name="fc1")(x)
		x = Dense(128, activation=activation, name="fc2")(x)
		x = Dense(classes, activation="softmax", name="prediction")(x)

		model = Model(input, x, name="VGG19_Net")

		model.summary()

		return model


	@staticmethod
	def train():

		print("\t Loading CIFAR-100 dataset ...")
		(train_d, train_l), (test_d, test_l) = cifar100.load_data()

		train_l = to_categorical(train_l, num_classes=100)
		test_l = to_categorical(test_l, num_classes=100)

		input_shape = train_d.shape[1:]

		train_d = train_d.astype("float32") / 255
		test_d = test_d.astype("float32") / 255

		model = VGG19_Net.build(input_shape= input_shape)
		model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0005), metrics=["accuracy"])

		if not os.path.exists("models"):
			os.mkdir("models")
		best_weights = "models/best_weights_VGG16_CIFAR100_{val_acc:.4f}.h5"

		save_best_model = ModelCheckpoint(best_weights, monitor="val_acc", verbose=1, save_best_only=True)
		reduce_lr = ReduceLROnPlateau(monitor="val_acc", factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
		early_stopping = EarlyStopping(monitor="val_acc", patience=200, verbose=1)
		# reduce_lr = ReduceLROnPlateau(monitor="val_acc", factor=0.8, patience=100, verbose=1)

		print("\t Start training ...")
		# train_history = model.fit(train_d, train_l, batch_size=64, epochs=1000, verbose=1, validation_data=(test_d, test_l),
		# 	                          callbacks=[reduce_lr, save_best_model, early_stopping])

		train_history = model.fit(train_d, train_l, batch_size=32, epochs=1000, verbose=1, validation_data=(test_d, test_l))


if __name__ == '__main__':
	VGG19_Net.train()