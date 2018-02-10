# How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow 1.5 (GPU) on Windows 10

## Brief Summary
This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Windows 10, 8, or 7. (It will also work on Linux-based OS with some minor changes.) I will also be releasing a YouTube video that walks through this tutorial.

This readme describes every step required to get going with your own object detection classifier: 
1. Installing TensorFlow-GPU
2. Setting up the Object Detection directory structure an Anaconda Virtual Environment
3. Gathering and labeling pictures
4. Generating training data
5. Creating a lap map and configuring training
6. Training
7. Exporting the inference graph
8. Testing and using your newly trained object detection classifier

The repository provides all the files needed to train a "Pinochle Deck" playing card detector that can accurately detect nines, tens, jacks, queens, kings, and aces. The tutorial describes how to replace these files with your own files to train a detection classifier for whatever your heart desires. It also has Python scripts to test your classifier out on an image, video, or webcam feed.

## Introduction
The purpose of this tutorial is to explain how to train your own convolutional neural network object detection classifier for multiple objects, starting from scratch. At the end of this tutorial, you will have a program that can identify and draw boxes around specific objects in pictures, videos, or in a webcam feed.

There are several good tutorials available for how to use TensorFlow’s Object Detection API to train a classifier for a single object. However, these usually assume you are using a Linux operating system. If you’re like me, you might be a little hesitant to install Linux on your high-powered gaming PC that has the sweet graphics card you’re using to train a classifier. The Object Detection API seems to have been developed on a Linux-based OS, so to set up TensorFlow to train a model on Windows, there are several workarounds that need to be used in place of commands that would work fine on Linux. Also, this tutorial provides instructions for training a classifier that can detect multiple objects, not just one.

The tutorial is written for Windows 10, and it will also work for Windows 7 and 8. The general procedure can also be used for Linux operating systems, but file paths and package installation commands will need to change accordingly. 

TensorFlow-GPU allows your PC to use the video card to provide extra processing power while training, so it will be used for this tutorial. In my experience, using TensorFlow-GPU instead of regular TensorFlow reduces training time by a factor of about 8 (3 hours to train instead of 24 hours). Regular TensorFlow can also be used for this tutorial, but it will take longer. If you use regular TensorFlow, you do not need to install CUDA and cuDNN in Step 1. I used TensorFlow-GPU v1.5 while writing this tutorial, but it will likely work for future versions of TensorFlow.


## Steps
### 1. Install TensorFlow-GPU 1.5 (skip this step if TensorFlow-GPU 1.5 is already installed)
Install TensorFlow-GPU by following the instructions in [this YouTube Video by Mark Jay](https://www.youtube.com/watch?v=RplXYjxgZbw).

The video is made for TensorFlow-GPU v1.4, but the “pip install --upgrade tensorflow-gpu” command will automatically download version 1.5. Download and install CUDA v9.0 and cuDNN v7.0 (rather than CUDA v8.0 and cuDNN v6.0 as instructed in the video), because they are supported by TensorFlow-GPU v1.5. As future versions of TensorFlow are released, you will likely need to continue updating the CUDA and cuDNN versions to the latest supported version.

Be sure to install Anaconda with Python 3.6 as instructed in the video, as the Anaconda virtual environment will be used for the rest of this tutorial.

Visit [TensorFlow's website](https://www.tensorflow.org/install/install_windows) for further installation details, including how to install it on other operating systems (like Linux). The [object detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) itself also has [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

### 2. Set up TensorFlow Directory and Anaconda Virtual Environment
The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model. 

This portion of the tutorial goes over the full set up required. It is fairly meticulous, but follow the instructions closely, because improper setup can cause unwieldy errors down the road.

#### 2a. Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in C: and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.
* Picture of directory here?

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into C:\tensorflow1 you just created. Rename “models-master” to just “models”.
* Picture of directory here?

(Note, this tutorial was done using this [GitHub commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99). If portions of this tutorial do not work, it may be necessary to download and use this exact commit rather than the most up-to-date version.)

#### 2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy. I initially started with the SSD-MobileNet-V1 model, but it didn’t do a very good job identifying the cards in my images. I re-trained my detector on the Faster-RCNN-Inception-V2 model, and the detection worked considerably better without a noticeable difference in speed.

You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational power (such as a smart phone or Raspberry Pi), use the SDD-MobileNet model. If you will be running your detector on a decently powered laptop or desktop PC, use one of the RCNN models. 

This tutorial will use the Faster-RCNN-Inception-V2 model. [Download the model here](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) Open the downloaded faster_rcnn_inception_v2_coco_2018_01_28.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the C:\tensorflow1\models\research\object_detection folder. (Note: The model date and version will likely change in the future, but it should still work with this tutorial.)

#### 2c. Download this tutorial's repository from GitHub
Download the full repository located on this page (scroll to the top and click Clone or Download) and extract all the contents directly into the C:\tensorflow1\models\research\object_detection directory. This establishes a specific directory structure that will be used for the rest of the tutorial. 

At this point, here is what your \object_detection folder should look like:

* Picture of directory

This repository contains the images, annotation data, .csv files, and TFRecords needed to train a "Pinochle Deck" playing card detector. You can use these images and data to practice making your own Pinochle Card Detector. It also contains Python scripts that are used to generate the training data, as well as test to test out the object detection classifier on images, videos, or a webcam feed. You can ignore the \doc folder and its files, they are just there to hold the images used for this readme.

If you want to practice training your own "Pinochle Deck" card detector, you can leave all the files as they are. You can follow along with this tutorial to see how each of the files were generated, and then run the training.

If you want to train your own object detector, delete the following files (do not delete the folders):
- All files in \object_detection\images\train and \object_detection\images\test
- The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
- All files in \object_detection\training
- The “train.record” and “test.record” files in \object_detection
-	All files in \object_detection\inference_graph

Now, you are ready to start from scratch in training your own object detector. This tutorial will assume that all the files listed above were deleted, and will go on to explain how to generate the files for your own training dataset.

