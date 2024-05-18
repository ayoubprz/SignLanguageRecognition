# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:15:48 2024

@author: dayoub
"""


import tensorflow as tf
from tensorflow.keras import layers

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import math

import time

num_classes = 46
num_students = 80
num_data =  num_classes * num_students
num_sensors = 7
num_data_per_sensor_per_second = 6
len_sensor_data = 80
y = np.arange(0, num_data)
y2 = y//num_students

from sklearn.model_selection import train_test_split


dataset = np.load('../01FinalDataset/editedDataset.npy')


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    
    
    def call(self, inputs):
        even_inputs = inputs[:,:, ::2,:]
        odd_inputs  = inputs[:,:, 1::2,:]

        # Add the corresponding elements of the even and odd outputs
        output = (even_inputs + odd_inputs)/math.sqrt(2)

        return output

def residual_block(inputs, filters, stride=1):
    shortcut = inputs
    x = layers.Conv2D(filters, kernel_size=(3,1), strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=(3,1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def create_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    X = CustomLayer()(inputs)
    x = layers.Conv2D(32, kernel_size=(2,1), strides=1, padding='same')(X)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    
    """x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    
    x = residual_block(x, filters=512, stride=2)
    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)"""
    
    x = layers.GlobalAveragePooling2D()(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Example usage:
input_shape = (6, 80, 7)


trainPortion = 0.80
cross_val_counter = 1


# xTrain, yTrain, xTest, yTest = loadDataset(trainPortion)
xTrain, xTest, yTrain, yTest = train_test_split(dataset, y2, test_size=0.2)

    
# xTrain = np.moveaxis(xTrain, -1, 1)
# xTrain42_80 = xTrain.reshape(xTrain.shape[0], 6,80)
# xTest = np.moveaxis(xTest, -1, 1)
# xTest42_80 = xTest.reshape(xTest.shape[0], 42,80)
    
resnet_model = create_resnet(input_shape, num_classes)
resnet_model.compile(optimizer='adam', 
                      loss = 'sparse_categorical_crossentropy', 
                      metrics=['accuracy'])



start_time = time.time()
net_hist = resnet_model.fit(xTrain, yTrain, batch_size=20, epochs=5)
print("training time is: ", time.time()-start_time)



test_labels = resnet_model.predict(xTest)
test_labels = np.argmax(test_labels, axis=1)
report = metrics.classification_report(yTest, test_labels)
m = metrics.confusion_matrix(yTest, test_labels)

FP = sum(m.sum(axis=0) - np.diag(m))
FN = sum(m.sum(axis=1) - np.diag(m))
TP = sum(np.diag(m))
TN = sum(m.sum() - (m.sum(axis=0) - np.diag(m) + m.sum(axis=1) - np.diag(m) + np.diag(m)))


mean_Accuracy = (TP + TN)/(TP+TN+FP+FN)
mean_Precision = TP/(TP+FP)
mean_Recall = TP/(TP+FN)


mean_F1Score = (2 * mean_Precision * mean_Recall)  /  (mean_Precision + mean_Recall)











