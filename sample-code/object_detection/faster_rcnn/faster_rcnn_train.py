
import random, sys, pprint, time, pickle, math, copy, os
from optparse import OptionParser
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

class Config:

	def __init__(self):

		self.verbose = True
		self.network = "vgg"

		self.horizontal_flips = False
		self.vertical_flips = False
		self.rotate_90 = False

		self.anchor_box_scales = [64, 128, 256]
		self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

		self.im_resize = 300

		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 1.0

		self.num_rois = 4

		self.rpn_stride = 16

		self.balanced_classes = False

		self.std_scaling = 4.0
		self.classifier_regressor_std = [8.0, 8.0, 4.0, 4.0]

		self.rpn_min_iou = 0.3
		self.rpn_max_iou = 0.7

		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		self.class_mapping = None
		self.model_path = None


def get_data(input_path):
	'''
	Parse data from annotation file
	:param input_path: annotation file path
	:return:
		all_data: list(filepath, width, height, list(bboxes))
		classes_count: dict{key- class_name: value- count_num} -- {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
		class_mapping: dict{key- class_name: value- idx} --  {'Car': 0, 'Mobile phone': 1, 'Person': 2}
	'''
	found_bg = False
	all_imgs = {}
	classes_count = {}
	class_mapping = {}
	visualise = True

	i = 1
	with open(input_path, "r") as f:
		print('Parsing annotation files')
		for line in f:
			sys.stdout.write("\r"+"idx="+str(i))
			i += 1
			line_split = line.strip().split(",")
			(filename, x1, y1, x2, y2, class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == "bg" and found_bg == False:
					print("class name is bg. Will be treated as background(hard negative minning).")
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}

				img = cv2.imread(filename)
				(rows, cols) = img.shape[:2]

				all_imgs[filename]["filepath"] = filename
				all_imgs[filename]["width"] = cols
				all_imgs[filename]["height"] = rows
				all_imgs[filename]["bboxes"] = []

			all_imgs[filename]["bboxes"].append({"class": class_name, "x1": int(x1), "x2": int(x2), "y1": int(y1), "y2": int(y2)})

		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])

		if found_bg:
			if class_mapping["bg"] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
				val_to_switch = class_mapping["bg"]
				class_mapping["bg"] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch

		return all_data, classes_count, class_mapping


class RoiPoolingConv(Layer):

	def __init__(self, pool_size, num_rois, **kwargs):

		self.dim_ordering = K.image_dim_ordering()
		self.pool_size = pool_size
		self.num_rois = num_rois

		super(RoiPoolingConv, self).__init__(**kwargs)

	def build(self, input_shape):
		self.nb_channels = input_shape[0][3]

	def compute_output_shape(self, input_shape):
		return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

	def call(self, x, mask=None):

		assert (len(x) == 2)

		img = x[0]
		rois = x[1]
		input_shape = K.shape(img)

		outputs = []

		for roi_idx in range(self.num_rois):

			x = rois[0, roi_idx, 0]
			y = rois[0, roi_idx, 1]
			w = rois[0, roi_idx, 2]
			h = rois[0, roi_idx, 3]

			'''
			to do 
			'''


def rpn_layer(base_layers, num_anchors):
	x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="normal", name="rpn_conv1")(base_layers)
	x_class = Conv2D(num_anchors, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="rpn_out_class")(x)
	x_regressor = Conv2D(num_anchors, (1, 1), activation="linear", kernel_initializer="zero", name="rpn_out_regessor")(x)
	return [x_class, x_regressor, base_layers]
