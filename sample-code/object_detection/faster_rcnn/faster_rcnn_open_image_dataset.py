import time, sys, os, random
import numpy as np
import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
from shutil import copyfile

import cv2
import tensorflow as tf

base_path = 'Dataset/Open Images Dataset v4 (Bounding Boxes)'
images_boxable_fname = 'train-images-boxable.csv'
annotations_bbox_fname = 'train-annotations-bbox.csv'
class_descriptions_fname = 'class-descriptions-boxable.csv'

images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
# images_boxable.head()
annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
# annotations_bbox.head()
class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname))
# class_descriptions.head()

