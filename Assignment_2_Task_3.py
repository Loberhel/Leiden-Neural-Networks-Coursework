
# coding: utf-8

# In[2]:

import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from functools import partial
from tensorflow.keras.optimizers import Adam, SGD


# # Task 3

# In[ ]:

#Import the data 
labels_data = np.load('labels.npy')
images_data = np.load('images.npy')

#FIXME NEED TO SPLIT TRAIN INTO A VALIDATION, TEST, and TRAIN SET

#split the data up into minute and hour 

labels_1D = []
for label in labels_data:
    labels_1D.append(60*label[0]+label[1])
    
labels_1D = np.asarray(labels_1D)

train_data, test_data = images_data[:9000], images_data[14400:]
train_labels, test_labels = labels_1D[:9000], labels_1D[3600:]

test_data = test_data.reshape(9000,150,150,1)
test_labels= test_labels.reshape(9000,1)

train_data = train_data.reshape(9000,150,150,1)
train_labels= train_labels.reshape(9000,1)


# In[ ]:

model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=(150,150,1)))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(1, activation="relu"))

model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')

history = model.fit(train_data, train_labels, batch_size=10, verbose=1, epochs=6, validation_data=(test_data, test_labels))
score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])


# 

# In[19]:

#We should make images greyscale to allow for faster CNN
#We should divide target minute value by 60 since CNN perform better on smaller ranges
def spliting_labels(labels):
    
    
    label_hour = []
    label_min = []
    for label in labels:
        
        label_hour.append(label[0])
        label_min.append(label[1]/60)
    
    label_hour = np.asarray(label_hour)
    label_min = np.asarray(label_min)

    return label_hour, label_min

#train_data, test_data = images_data[:9000], images_data[9000:]
#train_labels, test_labels = labels_data[:9000], labels_data[9000:]



# ## CNN Method
# 
# The idea here was to classify the minute and hour separately. The network has two outputs the hour and minute. Hour values can be 0 to 11 and the minutes can be 0 to 59. Can look at this from a regression stand point becuase we want to be as accurate in the minute value as possible. We use regression for the minute and classification for the hour. 
# 
# We first add Convolutional layers to the netwrok, which will extract significant features from the image. Then there are 2 branches of Fully-Connected layers. One branch is for finding the hour and one for finding the minute.
# 
# Since predicting hour value is a classification task. There would be 12 output nodes in the hour-branch. And we apply a Softmax activation on top of output nodes.
# 
# In the minute-branch, there would be just one output node with the Linear activation, since in regression we just need a single value. Linear activation is essentially no activation. I will not go into details of classification and regression here.
# 
# 

# In[ ]:


#shape= image size image size dimension 1,  size dimension 2, channel
inp = keras.layers.Input(shape=[150,150,1], batch_size=300)
                        
# Convolutional Layers
m = keras.layers.Convolution2D(16, kernel_size=3, strides=2, activation='relu')(inp)
m = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(m)
m = keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')(m)
m = keras.layers.MaxPooling2D(pool_size=(2, 2))(m)
m = keras.layers.Conv2D(128, kernel_size=3, strides=1, activation='relu')(m)
m = keras.layers.MaxPooling2D(pool_size=(2, 2))(m)
m = keras.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu')(m)
m = keras.layers.Dropout(.4)(m)
m = keras.layers.Flatten()(m)

# Hour branch with 12 nodes

hour = keras.layers.Dense(144, activation='relu')(m)
hour = keras.layers.Dense(144, activation='relu')(hour)
hour = keras.layers.Dense(12, activation='softmax', name='hour')(hour)

# Minute Branch has one output node with the Linear activation, since in regression we just need a single value. 
minute = keras.layers.Dense(100, activation='relu')(m)
minute = keras.layers.Dense(200, activation='relu')(minute)
minute = keras.layers.Dense(1, activation='linear', name='minute')(minute)

model = keras.models.Model(inputs=inp, outputs=[hour, minute])


model.summary()


# In[ ]:




# In[22]:

#Split the data into test and training set I just choose in half can change later
train_data_CNN, test_data_CNN = images_data[:9000], images_data[9000:]
train_labels_CNN , test_labels_CNN = labels_data[:9000], labels_data[9000:]

test_label_hour, test_label_min = spliting_labels(test_labels_CNN)
train_label_hour, train_label_min = spliting_labels(train_labels_CNN)

#print(train_label_hour.shape(9000,150,150,1))
train_data_CNN = train_data_CNN.reshape(9000,150,150,1)
train_label_hour = train_label_hour.reshape(9000,1)
train_label_min = train_label_min.reshape(9000,1)

test_data_CNN = test_data_CNN.reshape(9000,150,150,1)
test_label_hour = test_label_hour.reshape(9000,1)
test_label_min = test_label_min.reshape(9000,1)

adam = tf.keras.optimizers.Adam(lr=.00001)
model.compile(loss=['sparse_categorical_crossentropy', 'mse'], optimizer=adam, metrics=['accuracy', 'mse'])

history = model.fit(train_data_CNN, [train_label_hour, train_label_min], epochs=10, batch_size=300, validation_data=(test_data_CNN, [test_label_hour, test_label_min]))


# In[ ]:



