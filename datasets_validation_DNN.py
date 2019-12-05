# train a model using online dataset
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session()
import pydot

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import numpy as np
from DNN_Computation import build_model

inputs = layers.Input(shape=(784,), name='digits')
x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='sigmoid', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(60000, 784).astype('float32') / 255, x_test.reshape(10000, 784).astype('float32') / 255
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop())
model.fit(x_train, y_train, batch_size=64, epochs=1)
predictions = model.predict(x_test)

#np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

weights = model.get_weights()

# self-designed test model with trained weights above
#activation-nonlinear function(0:'linear', 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax') caution: activation is integer
activations = [1, 2, 4]
my_model = build_model(weights, activations, True)
my_predictions = my_model.computing(np.transpose(x_test))
my_predictions = np.transpose(my_predictions)
np.testing.assert_allclose(predictions, my_predictions, rtol=1e-6, atol=1e-6)

print(my_predictions[0])

print(predictions[0])
