import unittest
from myjson import json_decoders
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, concatenate, Masking  # noqa
from tensorflow.keras import Model  # noqa
from tensorflow.keras.models import load_model, Model  # noqa
import numpy as np
from KerasModelPadder import copy_weights
import os
from typing import Optional


def create_model(output_size, input_shape=100, middle_layer: Optional[Layer] = None):
    input = Input(shape=input_shape)
    if middle_layer:
        middle_layer = middle_layer(input)
    else:
        middle_layer = input
    output = Dense(output_size, kernel_initializer='random_uniform', activation='sigmoid')(middle_layer)
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

def set_weights_to_zero_and_copy(model_to_init: Model, path_to_load_weights_from: str) -> None:
    for layer in model_to_init.layers:
        set_weights_to_zero(layer)
    copy_weights(model_to_init, path_to_load_weights_from)


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

    def test_grow_lstm(self):
        model = create_model(1, input_shape=(7, 7), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        path = os.getcwd() + '/.pytest_cache/test_resize.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 6, 39) == 0
        model.save(path)

        model = create_model(1, input_shape=(7, 7), middle_layer=LSTM(20, kernel_initializer='random_uniform'))
        assert get_weight_value(model, 1, 6, 39) != 0
        assert get_weight_value(model, 1, 6, 79) != 0

        copy_weights(model, path)
        assert get_weight_value(model, 1, 6, 39) == 0
        assert get_weight_value(model, 1, 6, 79) != 0

    def test_truncate_lstm(self):
        model = create_model(1, input_shape=(7, 7), middle_layer=LSTM(20, kernel_initializer='random_uniform'))
        path = os.getcwd() + '/.pytest_cache/test_resize.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 6, 79) == 0
        assert get_weight_value(model, 1, 6, 39) == 0
        model.save(path)

        model = create_model(1, input_shape=(7, 7), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        assert get_weight_value(model, 1, 6, 39) != 0

        copy_weights(model, path)
        assert get_weight_value(model, 1, 6, 39) == 0
