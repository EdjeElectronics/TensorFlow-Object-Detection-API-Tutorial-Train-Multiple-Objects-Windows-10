# How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow 1.5 (GPU) on Windows 10

## Brief Summary
This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Windows 10, 8, or 7. (It will also work on Linux-based OS with some minor changes.)

This readme describes every step required to get going with your own object detection classifier: 
1. Installing TensorFlow-GPU
2. Setting up the Object Detection directory structure an Anaconda Virtual Environment
3. Gathering and labeling pictures
4. Generating training data
5. Creating a lap map and configuring training
6. Training
7. Exporting the inference graph
8. Testing and using your newly trained object detection classifier

The repository provides all the files needed to train a "Pinochle Deck" playing card detector that can accurately detect nines, tens, jacks, queens, kings, and aces. The tutorial describes how to replace these files with your own files to train a detection classifier for whatever your heart desires.

## Introduction
The purpose of this tutorial is to explain how to train your own convolutional neural network object detection classifier for multiple objects, starting from scratch. At the end of this tutorial, you will have a program that can identify and draw boxes around specific objects in pictures, videos, or in a webcam feed.

There are several good tutorials available for how to use TensorFlow’s Object Detection API to train a classifier for a single object. However, these usually assume you are using a Linux operating system. If you’re like me, you might be a little hesitant to install Linux on your high-powered gaming PC that has the sweet graphics card you’re using to train a classifier. The Object Detection API seems to have been developed on a Linux-based OS, so to set up TensorFlow to train a model on Windows, there are several workarounds that need to be used in place of commands that would work fine on Linux. Also, this tutorial provides instructions for training a classifier that can detect multiple objects, not just one.

The tutorial is written for Windows 10, and it will also work for Windows 7 and 8. The general procedure can also be used for Linux operating systems, but file paths and package installation commands will need to change accordingly. 

TensorFlow-GPU allows your PC to use the video card to provide extra processing power while training, so it will be used for this tutorial. In my experience, using TensorFlow-GPU instead of regular TensorFlow reduces training time by a factor of about 8 (3 hours to train instead of 24 hours). Regular TensorFlow can also be used for this tutorial, but it will take longer. If you use regular TensorFlow, you do not need to install CUDA and cuDNN in Step 1. I used TensorFlow-GPU v1.5 while writing this tutorial, but it will likely work for future versions of TensorFlow.
