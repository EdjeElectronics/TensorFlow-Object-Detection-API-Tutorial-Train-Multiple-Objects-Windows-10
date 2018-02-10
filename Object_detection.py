## Evan's very own Object Detection program using Tensforflow MobileNet-SSD model

## Some of this will be copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some will be copied from this guy's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I will change it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'card_inference_graph'
IMAGE_NAME = 'test3.jpg'


# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
#ckpt_path = 'C://Users/Evan/Documents/Object_Detection_stuff/tensorflow/models/research/object_detection/raccoon_inference_graph/frozen_inference_graph.pb'
#PATH_TO_CKPT = ckpt_path.encode('utf8')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','card-labelmap.pbtxt')
#PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'object-detection.pbtxt')
#label_path = 'C://Users/Evan/Documents/Object_Detection_stuff/tensorflow/models/research/object_detection/training/objectdetection.pbtxt'
#PATH_TO_LABELS = label_path.encode('utf8')
# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine

# EVAN YOU NEED TO LOOK AT THESE FILES AND FIGURE OUT WHAT THEY'RE DOING BECAUSE
# THIS SEEMS KIND OF EXCESSIVE

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
# EVAN, the with statement basically makes it all close down after it's done
# loading. Not sure what it does or means really.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

## EVAN, I think this section sort of defines what the outputs of the model
## are going to be

## Define input and output tensors for detection_graph

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

## Output tensors
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image, convert to RGB (which is needed by Tensorflow model), and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = image
image_rgb_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_rgb_expanded})

# Draw the results of the detection (aka 'visulaize the results')
## EVAN, you need to figure out what this STUPID FRICKIN visualization utility
## is doing so you can get rid of it

vis_util.visualize_boxes_and_labels_on_image_array(
    image_rgb,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8)

# All the results have been drawn on image_rgb. Convert back to BGR and display.
#final_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
cv2.imshow('Ayy', image_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

