import unittest
from myjson import json_decoders
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, concatenate, Masking  # noqa
from tensorflow.keras import Model  # noqa
from tensorflow.keras.models import load_model, Model  # noqa
import numpy as np
from KerasModelPadder import copy_weights
import os


def create_model(n):
    input = Input(shape=100)
    output = Dense(n, kernel_initializer='random_uniform', activation='sigmoid')(input)
    model = Model(inputs=[input], outputs=output)
    model.compile(loss='mean_squared_error',
                  optimizer='adam'
                  )

    return model

def set_weights_to_zero(layer: Layer) -> None:
    """ Sets the weights of the specified layer to zero """
    weight1_old = layer.weights[0]
    weight1_old_shape = weight1_old.shape
    weight1_new = np.zeros(weight1_old_shape)
    layer.weights[0].assign(weight1_new)

def get_weight_value(model: Model, layer_index: int, *indices: int) -> float:
    """ Gets the value of a particular weight in the model """
    weights = model.layers[layer_index].weights[0].numpy()

    if len(indices) != len(weights.shape):
        raise ValueError(f'Incorrect number of indices specified. Got {str(len(indices))}; expected {str(len(weights.shape))} (shape={str(weights.shape)})')
    return weights[indices]


class TestPadder(unittest.TestCase):
    def test_grow_dense(self):
        model = create_model(1)
        path = os.getcwd() + '/.pytest_cache/test_resize.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 99, 0) == 0
        model.save(path)

        model = create_model(2)
        assert get_weight_value(model, 1, 99, 0) != 0
        assert get_weight_value(model, 1, 99, 1) != 0
        
        copy_weights(model, path)
        assert get_weight_value(model, 1, 99, 0) == 0
        assert get_weight_value(model, 1, 99, 1) != 0

    def test_truncate_dense(self):
        model = create_model(2)
        path = os.getcwd() + '/.pytest_cache/test_resize.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 99, 0) == 0
        assert get_weight_value(model, 1, 99, 1) == 0
        model.save(path)

        model = create_model(1)
        assert get_weight_value(model, 1, 99, 0) != 0
        
        copy_weights(model, path)
        assert get_weight_value(model, 1, 99, 0) == 0
