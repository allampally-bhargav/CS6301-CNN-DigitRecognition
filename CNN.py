################################################################################
#
# LOGISTICS
#
#    <Your name as in eLearning> Bhargav Allampally
#    <Your UT Dallas identifier> BXA180005
#
# FILE
#
#    <CNN.py>
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. A summary of my cnn.py code:
#
#       <Forward path code summary / highlights>
#          1.  Performs a forward pass of the conv layer using the given input.
#              Returns a 3d numpy array with dimensions (h, w, num_filters) - input is a 2d numpy array
#       <Error code summary / highlights>
#          1. Used a Cross Entropy Loss function to calculate the loss
#
#
#    2. Accuracy display
#
#       <Per epoch display info cut and pasted from your training output>

#--- Epoch 1 ---
##[Step 100] Past 100 steps: Average Loss 2.209 | Accuracy: 21% | Time: 12 
##[Step 200] Past 100 steps: Average Loss 1.911 | Accuracy: 41% | Time: 24 
##[Step 300] Past 100 steps: Average Loss 1.341 | Accuracy: 64% | Time: 36 
##[Step 400] Past 100 steps: Average Loss 0.901 | Accuracy: 72% | Time: 48 
##[Step 500] Past 100 steps: Average Loss 0.863 | Accuracy: 74% | Time: 60 
##[Step 600] Past 100 steps: Average Loss 0.936 | Accuracy: 71% | Time: 72 
##[Step 700] Past 100 steps: Average Loss 0.813 | Accuracy: 75% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.547 | Accuracy: 82% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.743 | Accuracy: 76% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.721 | Accuracy: 82% | Time: 121 
#--- Epoch 2 ---
#[Step 100] Past 100 steps: Average Loss 0.605 | Accuracy: 81% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.610 | Accuracy: 79% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.576 | Accuracy: 82% | Time: 39 
#[Step 400] Past 100 steps: Average Loss 0.315 | Accuracy: 90% | Time: 51 
#[Step 500] Past 100 steps: Average Loss 0.510 | Accuracy: 84% | Time: 64 
#[Step 600] Past 100 steps: Average Loss 0.533 | Accuracy: 83% | Time: 76 
#[Step 700] Past 100 steps: Average Loss 0.533 | Accuracy: 86% | Time: 88 
#[Step 800] Past 100 steps: Average Loss 0.363 | Accuracy: 89% | Time: 100 
#[Step 900] Past 100 steps: Average Loss 0.593 | Accuracy: 82% | Time: 112 
#[Step 1000] Past 100 steps: Average Loss 0.521 | Accuracy: 87% | Time: 125 
#--- Epoch 3 ---
#[Step 100] Past 100 steps: Average Loss 0.462 | Accuracy: 83% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.457 | Accuracy: 86% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.490 | Accuracy: 86% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.230 | Accuracy: 95% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.420 | Accuracy: 86% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.393 | Accuracy: 88% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.443 | Accuracy: 85% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.303 | Accuracy: 92% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.517 | Accuracy: 84% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.431 | Accuracy: 91% | Time: 121 
#--- Epoch 4 ---
#[Step 100] Past 100 steps: Average Loss 0.380 | Accuracy: 87% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.380 | Accuracy: 89% | Time: 23 
#[Step 300] Past 100 steps: Average Loss 0.430 | Accuracy: 89% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.190 | Accuracy: 96% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.362 | Accuracy: 88% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.311 | Accuracy: 91% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.380 | Accuracy: 86% | Time: 84 
#[Step 800] Past 100 steps: Average Loss 0.273 | Accuracy: 93% | Time: 96 
#[Step 900] Past 100 steps: Average Loss 0.454 | Accuracy: 86% | Time: 108 
#[Step 1000] Past 100 steps: Average Loss 0.377 | Accuracy: 92% | Time: 120 
#--- Epoch 5 ---
#[Step 100] Past 100 steps: Average Loss 0.317 | Accuracy: 88% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.328 | Accuracy: 91% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.379 | Accuracy: 89% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.163 | Accuracy: 97% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.316 | Accuracy: 89% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.253 | Accuracy: 92% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.327 | Accuracy: 88% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.250 | Accuracy: 94% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.401 | Accuracy: 90% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.338 | Accuracy: 94% | Time: 121 
#--- Epoch 6 ---
#[Step 100] Past 100 steps: Average Loss 0.267 | Accuracy: 92% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.288 | Accuracy: 92% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.334 | Accuracy: 89% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.142 | Accuracy: 97% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.277 | Accuracy: 91% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.211 | Accuracy: 93% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.280 | Accuracy: 91% | Time: 84 
#[Step 800] Past 100 steps: Average Loss 0.229 | Accuracy: 95% | Time: 96 
#[Step 900] Past 100 steps: Average Loss 0.356 | Accuracy: 91% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.304 | Accuracy: 94% | Time: 121 
#--- Epoch 7 ---
#[Step 100] Past 100 steps: Average Loss 0.226 | Accuracy: 93% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.255 | Accuracy: 94% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.293 | Accuracy: 89% | Time: 39 
#[Step 400] Past 100 steps: Average Loss 0.125 | Accuracy: 98% | Time: 51 
#[Step 500] Past 100 steps: Average Loss 0.244 | Accuracy: 91% | Time: 63 
#[Step 600] Past 100 steps: Average Loss 0.178 | Accuracy: 97% | Time: 75 
#[Step 700] Past 100 steps: Average Loss 0.237 | Accuracy: 93% | Time: 88 
#[Step 800] Past 100 steps: Average Loss 0.208 | Accuracy: 95% | Time: 100 
#[Step 900] Past 100 steps: Average Loss 0.314 | Accuracy: 93% | Time: 112 
#[Step 1000] Past 100 steps: Average Loss 0.273 | Accuracy: 94% | Time: 124 
#--- Epoch 8 ---
#[Step 100] Past 100 steps: Average Loss 0.191 | Accuracy: 94% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.224 | Accuracy: 94% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.255 | Accuracy: 90% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.109 | Accuracy: 98% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.214 | Accuracy: 93% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.152 | Accuracy: 98% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.199 | Accuracy: 94% | Time: 84 
#[Step 800] Past 100 steps: Average Loss 0.187 | Accuracy: 95% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.277 | Accuracy: 94% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.243 | Accuracy: 94% | Time: 121 
#--- Epoch 9 ---
#[Step 100] Past 100 steps: Average Loss 0.160 | Accuracy: 94% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.197 | Accuracy: 94% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.221 | Accuracy: 91% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.096 | Accuracy: 98% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.187 | Accuracy: 94% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.132 | Accuracy: 99% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.168 | Accuracy: 96% | Time: 84 
#[Step 800] Past 100 steps: Average Loss 0.167 | Accuracy: 97% | Time: 96 
#[Step 900] Past 100 steps: Average Loss 0.246 | Accuracy: 95% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.214 | Accuracy: 95% | Time: 121 
#--- Epoch 10 ---
#[Step 100] Past 100 steps: Average Loss 0.133 | Accuracy: 94% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.171 | Accuracy: 94% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.189 | Accuracy: 95% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.083 | Accuracy: 99% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.164 | Accuracy: 95% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.115 | Accuracy: 99% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.140 | Accuracy: 97% | Time: 84 
#[Step 800] Past 100 steps: Average Loss 0.146 | Accuracy: 97% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.218 | Accuracy: 96% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.189 | Accuracy: 97% | Time: 121 
#--- Epoch 11 ---
#[Step 100] Past 100 steps: Average Loss 0.110 | Accuracy: 95% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.146 | Accuracy: 96% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.161 | Accuracy: 96% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.071 | Accuracy: 99% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.145 | Accuracy: 96% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.102 | Accuracy: 99% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.116 | Accuracy: 97% | Time: 84 
#[Step 800] Past 100 steps: Average Loss 0.125 | Accuracy: 97% | Time: 96 
#[Step 900] Past 100 steps: Average Loss 0.194 | Accuracy: 96% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.165 | Accuracy: 97% | Time: 121 
#--- Epoch 12 ---
#[Step 100] Past 100 steps: Average Loss 0.090 | Accuracy: 96% | Time: 11 
#[Step 200] Past 100 steps: Average Loss 0.124 | Accuracy: 96% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.135 | Accuracy: 96% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.060 | Accuracy: 99% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.128 | Accuracy: 98% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.090 | Accuracy: 99% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.096 | Accuracy: 99% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.106 | Accuracy: 98% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.171 | Accuracy: 96% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.144 | Accuracy: 97% | Time: 121 
#--- Epoch 13 ---
#[Step 100] Past 100 steps: Average Loss 0.074 | Accuracy: 96% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.103 | Accuracy: 97% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.112 | Accuracy: 96% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.050 | Accuracy: 99% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.115 | Accuracy: 98% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.079 | Accuracy: 99% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.079 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.089 | Accuracy: 99% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.150 | Accuracy: 97% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.125 | Accuracy: 97% | Time: 121 
#--- Epoch 14 ---
#[Step 100] Past 100 steps: Average Loss 0.061 | Accuracy: 96% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.085 | Accuracy: 98% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.093 | Accuracy: 97% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.042 | Accuracy: 100% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.103 | Accuracy: 98% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.069 | Accuracy: 99% | Time: 73 
#[Step 700] Past 100 steps: Average Loss 0.066 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.075 | Accuracy: 99% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.130 | Accuracy: 97% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.107 | Accuracy: 98% | Time: 121 
#--- Epoch 15 ---
#[Step 100] Past 100 steps: Average Loss 0.051 | Accuracy: 98% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.070 | Accuracy: 98% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.076 | Accuracy: 100% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.036 | Accuracy: 100% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.092 | Accuracy: 99% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.060 | Accuracy: 99% | Time: 72 
#[Step 700] Past 100 steps: Average Loss 0.056 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.063 | Accuracy: 100% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.110 | Accuracy: 97% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.091 | Accuracy: 98% | Time: 121 
#--- Epoch 16 ---
#[Step 100] Past 100 steps: Average Loss 0.042 | Accuracy: 99% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.058 | Accuracy: 98% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.063 | Accuracy: 100% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.031 | Accuracy: 100% | Time: 50 
#[Step 500] Past 100 steps: Average Loss 0.082 | Accuracy: 99% | Time: 64 
#[Step 600] Past 100 steps: Average Loss 0.052 | Accuracy: 99% | Time: 76 
#[Step 700] Past 100 steps: Average Loss 0.048 | Accuracy: 100% | Time: 89 
#[Step 800] Past 100 steps: Average Loss 0.055 | Accuracy: 100% | Time: 101 
#[Step 900] Past 100 steps: Average Loss 0.092 | Accuracy: 97% | Time: 113 
#[Step 1000] Past 100 steps: Average Loss 0.076 | Accuracy: 99% | Time: 125 
#--- Epoch 17 ---
#[Step 100] Past 100 steps: Average Loss 0.036 | Accuracy: 99% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.049 | Accuracy: 100% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.053 | Accuracy: 100% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.027 | Accuracy: 100% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.072 | Accuracy: 99% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.045 | Accuracy: 99% | Time: 73 
#[Step 700] Past 100 steps: Average Loss 0.041 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.047 | Accuracy: 100% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.074 | Accuracy: 98% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.062 | Accuracy: 99% | Time: 121 
#--- Epoch 18 ---
#[Step 100] Past 100 steps: Average Loss 0.031 | Accuracy: 99% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.041 | Accuracy: 100% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.045 | Accuracy: 100% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.024 | Accuracy: 100% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.064 | Accuracy: 99% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.039 | Accuracy: 100% | Time: 73 
#[Step 700] Past 100 steps: Average Loss 0.036 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.042 | Accuracy: 100% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.058 | Accuracy: 98% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.047 | Accuracy: 99% | Time: 121 
#--- Epoch 19 ---
#[Step 100] Past 100 steps: Average Loss 0.027 | Accuracy: 99% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.035 | Accuracy: 100% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.039 | Accuracy: 100% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.021 | Accuracy: 100% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.056 | Accuracy: 99% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.033 | Accuracy: 100% | Time: 73 
#[Step 700] Past 100 steps: Average Loss 0.031 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.036 | Accuracy: 100% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.045 | Accuracy: 99% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.036 | Accuracy: 100% | Time: 121 
#--- Epoch 20 ---
#[Step 100] Past 100 steps: Average Loss 0.024 | Accuracy: 99% | Time: 12 
#[Step 200] Past 100 steps: Average Loss 0.030 | Accuracy: 100% | Time: 24 
#[Step 300] Past 100 steps: Average Loss 0.033 | Accuracy: 100% | Time: 36 
#[Step 400] Past 100 steps: Average Loss 0.018 | Accuracy: 100% | Time: 48 
#[Step 500] Past 100 steps: Average Loss 0.048 | Accuracy: 99% | Time: 60 
#[Step 600] Past 100 steps: Average Loss 0.028 | Accuracy: 100% | Time: 73 
#[Step 700] Past 100 steps: Average Loss 0.027 | Accuracy: 100% | Time: 85 
#[Step 800] Past 100 steps: Average Loss 0.032 | Accuracy: 100% | Time: 97 
#[Step 900] Past 100 steps: Average Loss 0.037 | Accuracy: 100% | Time: 109 
#[Step 1000] Past 100 steps: Average Loss 0.029 | Accuracy: 100% | Time: 121 



#       <Final accuracy cut and pasted from your training output>
    # Accuracy: 0.0864

    
#
#    3. Performance display
#
#       <Total training time cut and pasted from your training output>
#       Total time spent:2455.48s
#       <Per layer info (type, input size, output size, parameter size, MACs, ...)>
#
################################################################################
################################################################################
#
# IMPORT
#
################################################################################

#
# you should not need any import beyond the below
# PyTorch, TensorFlow, ... is not allowed
#

import os,struct
from array import array as pyarray
import numpy as np
import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

#
# add other hyper parameters here with some logical organization
#

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'


# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)

train_data_cp = train_data

train_data        = train_data.reshape(DATA_NUM_TRAIN,1, DATA_ROWS, DATA_COLS) 
# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data_cp = test_data 
test_data        = test_data.reshape(DATA_NUM_TEST,1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
print(train_data.shape)   # (60000, 1, 28, 28)
print(train_labels.shape) # (60000,)
print(test_data.shape)    # (10000, 1, 28, 28)
print(test_labels.shape)  # (10000,)

################################################################################

#normalize data
train_data = train_data_cp
test_data = test_data_cp
#one hot encoded labels
# nb_classes = 10

# train_labels_one_hot = np.eye(nb_classes)[train_labels]

# test_labels_one_hot = np.eye(nb_classes)[test_labels]

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None

class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output

  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape

    input = input.flatten()
    self.last_input = input

    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals

    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
#-> 13x13x16
conv2 = Conv3x3(32)                  #11x11x16 -> 6x6x32
pool2 = MaxPool2()

import time
train_images = train_data.reshape((DATA_NUM_TRAIN, 28,28))
# Considering only 1000 labels per epoch as execution time is taking more than 15hours for 10 epochs
train_labels = train_labels[:1000]
test_images = test_data.reshape((DATA_NUM_TEST,28,28))
test_labels = test_labels[:1000]

#The output is 26x26x16 and not 28x28x16 due to taking 0 padding, 
#which decreases the input’s width and height by 2.
conv = Conv3x3(16)                  # 28x28x1 -> 26x26x16
#Each of the 16 filters in the conv layer produces a 26x26 output, so stacked together they make up a 26x26x16 volume. 
#this happens because of 3 × 3(filter size) × 16 (number of filters) = 144 weights
pool = MaxPool2()                  # 26x26x16 -> 13x13x16

softmax = Softmax(13 * 13 * 16, 10) # 13x13x16 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

print('MNIST CNN initialized!')
stime = time.time()
# Train the CNN for 3 epochs
epochs = 20
train_accuracy = []

for epoch in range(epochs):
  print('--- Epoch %d ---' % (epoch + 1))
  acc_counter = 0
  t = time.time()

  # Train!
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    # print(im.shape)
    if i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%% | Time: %d ' %
        (i + 1, loss / 100, num_correct, time.time()-t)
      )
      acc_counter = num_correct
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc
  train_accuracy.append(acc_counter)

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
test_accuracy = []
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc
  test_accuracy.append(num_correct)

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

print("\nTotal time spent:{0:.2f}s".format(time.time()-stime))

#plot of accuracy vs epoch
plt.figure(figsize=(12,9))

plt.plot(list(range(1, epochs+1)),train_accuracy )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN accuracy as epoch iterates')
plt.show()

