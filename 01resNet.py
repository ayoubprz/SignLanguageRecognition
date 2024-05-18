# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:57:49 2024

@author: dayoub
"""
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split


dataset = np.load('./datasetName.npy')


num_classes = 46
num_students = 80
num_data =  num_classes * num_students
num_sensors = 7
num_data_per_sensor_per_second = 6
len_sensor_data = 80
y = np.arange(0, num_data)
y2 = y//num_students


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
    
    x = layers.Conv2D(32, kernel_size=(2,1), strides=1, padding='same')(inputs)
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


input_shape = (6, 80, 7)


num_cross_val = 10

m = np.zeros((num_cross_val, num_classes, num_classes))

for i in range(num_cross_val):
        X_train, X_test, y_train, y_test = train_test_split(dataset, y2, test_size=0.2)
        resnet_model = create_resnet(input_shape, num_classes)
        resnet_model.compile(optimizer='adam', 
                             loss = 'sparse_categorical_crossentropy', 
                             metrics=['accuracy'])
        
        net_hist = resnet_model.fit(X_train, y_train, batch_size=5, epochs=10, validation_split=0.2)
        test_labels = resnet_model.predict(X_test)
        test_labels = np.argmax(test_labels, axis=1)
        #report = metrics.classification_report(y_test, test_labels)
        m[i] = metrics.confusion_matrix(y_test, test_labels)




zTP = np.zeros((num_cross_val, 46))
zFP = np.zeros((num_cross_val, 46))
zFN = np.zeros((num_cross_val, 46))
zTN = np.zeros((num_cross_val, 46))
for i in range(num_cross_val):
    zFP[i] = m[i].sum(axis=0) - np.diag(m[i])
    zFN[i] = m[i].sum(axis=1) - np.diag(m[i])
    zTP[i] = np.diag(m[i])
    zTN[i] = m[i].sum() - (zFP[i] + zFN[i] + zTP[i])

sTP = sum(sum(zTP))
sFP = sum(sum(zFP))
sFN = sum(sum(zFN))
sTN = sum(sum(zTN))

mean_Accuracy = (sTP + sTN)/(sTP+sTN+sFP+sFN)
mean_Precision = sTP/(sTP+sFP)
mean_Recall = sTP/(sTP+sFN)


mean_F1Score = (2 * mean_Precision * mean_Recall)  /  (mean_Precision + mean_Recall)


import matplotlib.pyplot as plt
plt.plot(net_hist.history['accuracy'])
plt.plot(net_hist.history['loss'])
plt.plot(net_hist.history['val_loss'])
plt.plot(net_hist.history['val_accuracy'])
plt.title('model accuracy and loss')
plt.ylabel('accuracy and loss')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Loss', 'val_loss', 'val_accuracy'], loc=7)
plt.xlim([0, 9])
plt.show()




