from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
prediction = model.predict(test_images)

model_config = model.get_config() # information of the model
weights = model.get_weights() # training parameters of layers

layers = model_config['layers'] # list of information for each layers


import CNN_Computation as CCN
import useful_functions as USF

MY_CNN_Model, MY_DNN_Model = CCN.build_model(layers, weights)

small_test = test_images[:100]
Conv_Result = MY_CNN_Model.computing(small_test)
Dense_Input = USF.Flatten(Conv_Result)
my_prediction = MY_DNN_Model.computing(np.transpose(Dense_Input))

np.testing.assert_allclose(prediction[:100], np.transpose(my_prediction), rtol=1e-6, atol=1e-6)




