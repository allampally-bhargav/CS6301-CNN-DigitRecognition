################################################################################
#
# LOGISTICS
#
#    <Your name as in eLearning> Bhargav Allampally
#    <Your UT Dallas identifier> BXA180005
#
# FILE
#
#    <nn.py >
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
#    1. A summary of my nn.py code:
#
#       <Forward path code summary / highlights>
#           1. we use this function to predict value of a training sample data.We perform forward pass on the training sample and 
#              calculate the output value using input layer activations
#       <Error code summary / highlights>
#           1. Used Mean Square error loss function to calculate the loss
#       <Backward path code summary / highlights>
#           1.  This is the backpropagation algorithm, for calculating the updates of the neural network's parameters based on the error obtained.
#           2. Error is calculated by comparing predicted value with the actual label
#       <Weight update code summary / highlights>
#           1.  Update network parameters according to update rule from Stochastic Gradient Descent.
#              θ = θ - η * ∇J(x, y), 
#              theta θ: a network parameter (e.g. a weight w)
#              eta η:   the learning rate
#           gradient ∇J(x, y):  the gradient of the objective function,
#                               i.e. the change for a specific theta θ
#
#    2. Accuracy display
#
#       <Per epoch display info cut and pasted from your training output>


#       Epoch: 1, Time Spent: 86.45s, Training Loss: 0.0450 Testing Accuracy: 38.8400
#       Epoch: 2, Time Spent: 173.26s, Training Loss: 0.0447 Testing Accuracy: 40.7990
#       Epoch: 3, Time Spent: 258.88s, Training Loss: 0.0446 Testing Accuracy: 56.2510
#       Epoch: 4, Time Spent: 344.37s, Training Loss: 0.0450 Testing Accuracy: 74.8330
#       Epoch: 5, Time Spent: 429.80s, Training Loss: 0.0450 Testing Accuracy: 82.4810
#       Epoch: 6, Time Spent: 515.21s, Training Loss: 0.0450 Testing Accuracy: 85.4720
#       Epoch: 7, Time Spent: 600.95s, Training Loss: 0.0450 Testing Accuracy: 86.7400
#       Epoch: 8, Time Spent: 686.50s, Training Loss: 0.0450 Testing Accuracy: 87.4780

#       <Final accuracy cut and pasted from your training output>

#       87.478

#    3. Performance display
#
#       <Total training time cut and pasted from your training output>
        # Total time spent:686.51s

#       <Per layer info (type, input size, output size, parameter size, MACs, ...)>
#      ------------------------------------------------------------------------------------------------------------------------------------------------------

#|								Per Layer Infor (type, input size, output size, parameter size, MACs..)												|

#------------------------------------------------------------------------------------------------------------------------------------------------------

#|	Division					      |			1 * 28 * 28			|		1 * 28 * 28			|			None					    |		28 * 28		  |

#|	Vectorization			      |			1 * 28 * 28			|		1 * 784				  |			None					    |		None		    |

#|	Matrix Multiplication		|			1 * 784				  |		1 * 1000			  |			784 * 1000				|		784 * 1000  |

#|	Addition					      |			1 * 1000			  |		1 * 1000			  |			1 * 1000				  |		1000		    |

#|	RELU						        |			1 * 1000			  |		1 * 1000			  |			None					    |		1000		    |

#|	Matrix Multiplication		|			1 * 1000			  |		1 * 100				  |			1000 * 1000				|		1000 * 100	|

#|	Addition					      |			1 * 100				  |		1 * 100				  |			1 * 100					  |		100			    |

#|	RELU						        |			1 * 100				  |		1 * 100				  |			None					    |		100			    |

#|	Matrix Multiplication		|			1 * 100				  |		1 * 10				  |			100 * 10				  |		100 * 10	  |

#|	Addition					      |			1 * 10				  |		1 * 10				  |			1 * 10					  |		10			    |

#|	Softmax						      |			1 * 10				  |		1 * 10				  |			None					    |		10 * 10		  |

#------------------------------------------------------------------------------------------------------------------------------------------------------

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
import urllib.request
import os.path
import gzip
import math
import numpy as np
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
train_data = (train_data/255.0)
test_data = (test_data/255.0)

import time
class NeuralNetwork:
  def __init__(self, layers, epochs=10, l_rate=0.001):
    self.layers = layers
    self.epochs = epochs
    self.l_rate = l_rate

    # we save all parameters in the neural network in this dictionary
    self.parameters = self.initialization()

  def initialization(self):
    # number of nodes in each layer
    input_layer=self.layers[0]  #number of neurons in layer1 - 784
    hidden_layer1=self.layers[1]  #number of neurons in hidden layer 1 - 1000
    hidden_layer2=self.layers[2]  #number of neurons in hidden layer 2 - 100
    output_layer=self.layers[3]  #number of neurons in output layer - 10

    parameters = {
        'W1':np.random.randn(hidden_layer1, input_layer) * np.sqrt(1. / hidden_layer1),
        'W2':np.random.randn(hidden_layer2, hidden_layer1) * np.sqrt(1. / hidden_layer2),
        'W3':np.random.randn(output_layer, hidden_layer2) * np.sqrt(1. / output_layer)
    }

    return parameters

  #Relu function provides faster computation than sigmoid function.Even computation of derivative is faster compared to sigmoid.
  def relu(self, x,derivative=False):
    if derivative:
      np.where(x>0,x,0)
    return np.clip(x,0,None)

  def forward_pass(self, x_train):
    '''we use this function to predict value of a training sample data'''
    parameters = self.parameters

    # input layer activations becomes sample
    parameters['A0'] = x_train

    # input layer to hidden layer 1
    parameters['Z1'] = np.dot(parameters["W1"], parameters['A0'])
    parameters['A1'] = self.relu(parameters['Z1'])

    # hidden layer 1 to hidden layer 2
    parameters['Z2'] = np.dot(parameters["W2"], parameters['A1'])
    parameters['A2'] = self.relu(parameters['Z2'])

    # hidden layer 2 to output layer
    parameters['Z3'] = np.dot(parameters["W3"], parameters['A2'])
    parameters['A3'] = self.softmax(parameters['Z3'])

    return parameters['A3']

  #Normalization function
  def softmax(self, x):
      # Numerically stable with large exponentials
      exps = np.exp(x - x.max())
      return exps / np.sum(exps, axis=0)


  def compute_loss(self,Y,Y_hat):
    L = 1 / 2 * np.mean((Y - Y_hat)**2)
    return L

    
    

  def backward_pass(self, y_train, result):
    '''
        This is the backpropagation algorithm, for calculating the updates
        of the neural network's parameters.
    '''
    parameters = self.parameters
    change_w = {}

    # Calculate W3 update
    #calculate the error by comparing predicted value and the expected y value
    error = result - y_train
    change_w['W3'] = np.dot(error, parameters['A3'])

    # Calculate W2 update
    error = np.multiply( np.dot(parameters['W3'].T, error), self.relu(parameters['Z2'], derivative=True) )
    change_w['W2'] = np.dot(error, parameters['A2'])

    # Calculate W1 update
    error = np.multiply( np.dot(parameters['W2'].T, error), self.relu(parameters['Z1'], derivative=True) )
    change_w['W1'] = np.dot(error, parameters['A1'])

    return change_w

  
  def train(self, x_train, y_train, x_val, y_val):
    '''we perform forward pass followed by the backward pass function and updating the network parameters.Then we pass on the test data to calculate accuracy'''
    start_time = time.time()
    train_loss_list = []
    test_loss_list = []
   # train_accuracy_list = []
    test_accuracy_list = []
    time_list = []
    for iteration in range(self.epochs):
        for x,y in zip(x_train, y_train):
            result = self.forward_pass(x)
            changes_to_w = self.backward_pass(y, result)
            self.update_network_parameters(changes_to_w)
            loss = self.compute_loss(y, result)
        accuracy,test_loss,predictions = self.compute_accuracy(x_val, y_val)
       # train_accuracy,train_loss = self.compute_accuracy(x_train,y_train)
        print('Epoch: {0}, Time Spent: {1:.2f}s, Training Loss: {2:.4f} Testing Accuracy: {3:.4f}'.format(
            iteration+1, time.time() - start_time, loss, accuracy
        ))
        test_accuracy_list.append(accuracy)
        #train_accuracy_list.append(train_accuracy)
        train_loss_list.append(loss)
        test_loss_list.append(test_loss)
        time_list.append(time.time() - start_time)
    return test_accuracy_list,train_loss_list,test_loss_list,time_list,predictions

  def update_network_parameters(self, changes_to_w):
    '''
        Update network parameters according to update rule from
        Stochastic Gradient Descent.

        θ = θ - η * ∇J(x, y), 
            theta θ:            a network parameter (e.g. a weight w)
            eta η:              the learning rate
            gradient ∇J(x, y):  the gradient of the objective function,
                                i.e. the change for a specific theta θ
    '''
    
    for key, value in changes_to_w.items():
        for w_arr in self.parameters[key]:
            w_arr -= self.l_rate * value

  def compute_accuracy(self, x_val, y_val):
    '''
        This function does a forward pass of x, then checks if the indices
        of the maximum value in the output equals the indices in the label
        y. Then it sums over each prediction and calculates the accuracy.
    '''
    predictions = []
    predictions1 = []

    for x, y in zip(x_val, y_val):
        output = self.forward_pass(x)
        test_loss = self.compute_loss(output,y_val)
        pred = np.argmax(output)
        predictions1.append(pred)
        predictions.append(pred == y)
    
    summed = sum(pred for pred in predictions) / 100.0
    return np.average(summed),test_loss,predictions1

#reshape the data for nn
train_reshape = train_data.reshape(-1,DATA_ROWS*DATA_COLS)
test_reshape = test_data.reshape(-1,DATA_ROWS*DATA_COLS)

#one hot encoded labels
encoded = 10

train_labels_one_hot = np.eye(encoded)[train_labels]

test_labels_one_hot = np.eye(encoded)[test_labels]

epoch = 8

#intialize NN with layers with neuron counts, epochs, learning rate
traditional_nn = NeuralNetwork(layers = [784, 1000, 100, 10], epochs=epoch, l_rate=0.001)
st = time.time()
test_accuracy_list,train_loss_list,test_loss_list,time_list,predictions = traditional_nn.train(train_reshape , train_labels_one_hot, test_reshape, test_labels_one_hot)

print("\nTotal time spent:{0:.2f}s".format(time.time()-st))

#per epoch display (epoch,time spent on each epoch,training loss,testing accuracy)

#Accuracy Display - list of accuracies
test_accuracy_list

#final value
max(test_accuracy_list)

#plot of accuracy vs epoch
plt.figure(figsize=(12,9))
plt.plot(list(range(1, epoch+1)),test_accuracy_list )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy as epoch iterates')
plt.show()



#Performance Display

plt.plot([1,2,3,4,5,6,7,8],train_loss_list,'r--',marker='o',label='train loss')
plt.plot([1,2,3,4,5,6,7,8],test_loss_list,'b--',marker='v',label='test loss')
plt.legend(['train loss','test loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()



print("------------------------------------------------------------------------------------------------------------------------------------------------------\n")
print("|								Per Layer Infor (type, input size, output size, parameter size, MACs..)												|\n")
print("------------------------------------------------------------------------------------------------------------------------------------------------------\n")
print("|	Division					      |			1 * 28 * 28			|		1 * 28 * 28			|			None					    |		28 * 28		  |\n")
print("|	Vectorization			      |			1 * 28 * 28			|		1 * 784				  |			None					    |		None		    |\n")
print("|	Matrix Multiplication		|			1 * 784				  |		1 * 1000			  |			784 * 1000				|		784 * 1000  |\n")
print("|	Addition					      |			1 * 1000			  |		1 * 1000			  |			1 * 1000				  |		1000		    |\n")
print("|	RELU						        |			1 * 1000			  |		1 * 1000			  |			None					    |		1000		    |\n")
print("|	Matrix Multiplication		|			1 * 1000			  |		1 * 100				  |			1000 * 1000				|		1000 * 100	|\n")
print("|	Addition					      |			1 * 100				  |		1 * 100				  |			1 * 100					  |		100			    |\n")
print("|	RELU						        |			1 * 100				  |		1 * 100				  |			None					    |		100			    |\n")
print("|	Matrix Multiplication		|			1 * 100				  |		1 * 10				  |			100 * 10				  |		100 * 10	  |\n")
print("|	Addition					      |			1 * 10				  |		1 * 10				  |			1 * 10					  |		10			    |\n")
print("|	Softmax						      |			1 * 10				  |		1 * 10				  |			None					    |		10 * 10		  |\n")
print("------------------------------------------------------------------------------------------------------------------------------------------------------\n")

