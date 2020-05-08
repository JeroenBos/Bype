import unittest
from myjson import json_decoders
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Masking  # noqa
from tensorflow.keras import Model  # noqa
from tensorflow.keras.models import load_model, Model  # noqa
import numpy as np
from KerasModelPadder import copy_weights


def create_model(n):
    input = Input(shape=100)
    output = Dense(n, kernel_initializer='random_uniform', activation='sigmoid')(input)
    model = Model(inputs=[input], outputs=output)
    model.compile(loss='mean_squared_error',
                  optimizer='adam'
                  )

    return model


class TestPadder(unittest.TestCase):
    def test_resize(self):
        model = create_model(1)
        serialized = model.to_json()
        print(serialized)
        path = '/home/jeroen/git/bype/python/tests/testweights.h5'

        # set to zeroes:
        layer1 = model.layers[1]
        weight1_old = model.layers[1].weights[0]
        weight1_old_shape = weight1_old.shape
        weight1_new = np.zeros(weight1_old_shape)
        layer1.weights[0].assign(weight1_new)

        model.save(path)

        model = create_model(2)
        copy_weights(model, path)

        # layer1 = reloaded.layers[1]
        # weight1_old = reloaded.layers[1].weights[0]
#
        # weights: list of Numpy arrays to set as initial weights. The list should have 2 elements, of 
        # shape (input_dim, output_dim) and (output_dim,) for weights and biases respectively.
#
        # weights = [np.zeros([692, 50]), np.zeros(50)]

        print(model.layers[1].weights[0].numpy())
