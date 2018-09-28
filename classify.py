import tensorflow as tf
import tensorflow.contrib.slim as slim  # TensorFlow-Slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import math
import os
import time
import pickle

GRAYSCALE = False
NUM_CLASSES = 43

def rgb_to_gray(images):
    """
    Convert batch of RGB images to grayscale
    Use simple average of R, G, B values, not weighted average

    Arguments:
        * Batch of RGB images, tensor of shape (batch_size, 32, 32, 3)

    Returns:
        * Batch of grayscale images, tensor of shape (batch_size, 32, 32, 1)
    """
    images_gray = np.average(images, axis=3)
    images_gray = np.expand_dims(images_gray, axis=3)
    return images_gray
#
# def preprocess_data(X, y):
#     """
#     Preprocess image data, and convert labels into one-hot
#
#     Arguments:
#         * X: Image data
#         * y: Labels
#
#     Returns:
#         * Preprocessed X, one-hot version of y
#     """
#     # Convert from RGB to grayscale if applicable
#     if GRAYSCALE:
#         X = rgb_to_gray(X)
#
#     # Make all image array values fall within the range -1 to 1
#     # Note all values in original images are between 0 and 255, as uint8
#     X = X.astype('float32')
#     X = (X - 128.) / 128.
#
#     # Convert the labels from numerical labels to one-hot encoded labels
#     y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
#     for i, onehot_label in enumerate(y_onehot):
#         onehot_label[y[i]] = 1.
#     y = y_onehot
#
#     return X, y


####################### Data Loading, display , etc. ###################################

train_file = '/home/deep_learning/rahul/tfclassifier/dataset/train.p'
test_file = '/home/deep_learning/rahul/tfclassifier/dataset/test.p'

#loading of pickled data and check the data###################
with open(train_file,'rb') as f:
    train = pickle.load(f)
with open(test_file,'rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

num_train = X_train.shape[0]
num_test = X_test.shape[0]
image_shape = X_train.shape[1:3]
num_classes = np.unique(y_train).shape[0]

print("Number of training examples =", num_train)
print("Number of testing examples =", num_test)
print("Image data shape =", image_shape)
print("Number of classes =", num_classes)


############## Data display################
indices = np.random.choice(list(range(num_train)), size=3, replace=False)
print(indices)
for index in indices:
    image = X_train[index]
    labels = y_train[index]
    #print(image)
    im = Image.fromarray(image)
    im.show()
    cv2.waitKey(1)

################class label across training data############

labels, count = np.unique(y_train, return_counts=True)

# Plot the histogram
plt.rcParams["figure.figsize"] = [15, 5]
axes = plt.gca()
axes.set_xlim([-1,43])

plt.bar(labels, count)#, tick_label=labels, width=0.8, align='center')
plt.title('Class Distribution across Training Data')
plt.show()

#############class label across test data#############

labels, count = np.unique(y_test, return_counts=True)

# Plot the histogram
plt.rcParams["figure.figsize"] = [15, 5]
axes = plt.gca()
axes.set_xlim([-1,43])

plt.bar(labels, count)#, tick_label=labels, width=0.8, align='center')
plt.title('Class Distribution across Test Data')
plt.show()
