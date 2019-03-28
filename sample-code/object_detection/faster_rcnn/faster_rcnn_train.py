
import random, sys, pprint, time, pickle, math, copy, os
from optparse import OptionParser
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

from

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
		classes_count: dict{key- class_name: value- count_num} -- {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
		class_mapping: dict{key- class_name: value- idx} --  {'Car': 0, 'Mobile phone': 1, 'Person': 2}
	'''