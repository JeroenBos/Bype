import unittest
from myjson import json_decoders
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, concatenate  # noqa
from tensorflow.keras import Model  # noqa
from tensorflow.keras.models import load_model, Model  # noqa
import numpy as np
from KerasModelPadder import copy_weights, min_dim
import os
from typing import Optional, Union



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

def set_weights_to_zero(layer_or_model: Union[Layer, Model]) -> None:
    """ Sets the weights of the specified layer or model to zero """

    if isinstance(layer_or_model, Model):
        for layer in layer_or_model.layers:
            set_weights_to_zero(layer)
        return

    layer: Layer = layer_or_model
    for weights_set in layer.weights:
        weight1_new = np.zeros(weights_set.shape)
        weights_set.assign(weight1_new)

def print_layer_shapes(model: Model) -> None:
    for i, layer in enumerate(model.layers):
        if len(layer.weights) == 0:
            print(f'{i}: None, bias: None')
        else:
            print(f'{i}: {layer.weights[0].shape}, bias: {layer.weights[1].shape}')

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

def compare_predictions(model_a: Model, model_b: Model) -> None:
    assert model_a.layers[0].input_shape == model_b.layers[0].input_shape

    input_shape = model_a.layers[0].input_shape[0][1:]  # 1 and [1:] replace None by 1
    dummy_input = np.random.rand(1, *input_shape)  

    output_shapes = [(1, *model.layers[-1].output_shape[1:]) for model in [model_a, model_b]]
    truncate_shape = min_dim(*output_shapes)

    y_a = model_a.predict(dummy_input).resize(*truncate_shape)
    y_b = model_b.predict(dummy_input).resize(*truncate_shape)

    assert y_a == y_b

class TestPadder(unittest.TestCase):
    def test_grow_dense(self):
        model = create_model(1)
        path = os.getcwd() + '/.pytest_cache/test_grow_dense.h5'

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
        path = os.getcwd() + '/.pytest_cache/test_truncate_dense.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 99, 0) == 0
        assert get_weight_value(model, 1, 99, 1) == 0
        model.save(path)

        model = create_model(1)
        assert get_weight_value(model, 1, 99, 0) != 0

        copy_weights(model, path)
        assert get_weight_value(model, 1, 99, 0) == 0

    def test_grow_lstm(self):
        model = create_model(1, input_shape=(1, 7), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        path = os.getcwd() + '/.pytest_cache/test_grow_lstm.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 6, 39) == 0
        model.save(path)

        model = create_model(1, input_shape=(1, 7), middle_layer=LSTM(20, kernel_initializer='random_uniform'))
        assert get_weight_value(model, 1, 6, 39) != 0
        assert get_weight_value(model, 1, 6, 79) != 0

        copy_weights(model, path)
        assert get_weight_value(model, 1, 6, 39) == 0
        assert get_weight_value(model, 1, 6, 79) != 0

    def test_truncate_lstm(self):
        model = create_model(1, input_shape=(1, 7), middle_layer=LSTM(20, kernel_initializer='random_uniform'))
        path = os.getcwd() + '/.pytest_cache/test_truncate_lstm.h5'

        set_weights_to_zero(model.layers[1])
        assert get_weight_value(model, 1, 6, 79) == 0
        assert get_weight_value(model, 1, 6, 39) == 0
        model.save(path)

        model = create_model(1, input_shape=(1, 7), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        assert get_weight_value(model, 1, 6, 39) != 0

        copy_weights(model, path)
        assert get_weight_value(model, 1, 6, 39) == 0

    def test_idempotent_prediction_test(self):
        model = create_model(1)
        compare_predictions(model, model)


    def test_dense_grown_with_zeroes_predicts_same(self):
        path = os.getcwd() + '/.pytest_cache/test_dense_grown_with_zeroes_predicts_same.h5'

        original_model = create_model(1)
        original_model.save(path)

        reload_model = create_model(2)
        set_weights_to_zero(reload_model)

        copy_weights(reload_model, path)

        compare_predictions(original_model, reload_model)

    def test_lstm_grown_to_dense_with_zeroes_predicts_same(self):
        path = os.getcwd() + '/.pytest_cache/test_lstm_grown_to_dense_with_zeroes_predicts_same.h5'

        original_model = create_model(1, input_shape=(1, 7), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        original_model.save(path)
        print_layer_shapes(original_model)
        reload_model = create_model(2, input_shape=(1, 7), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        set_weights_to_zero(reload_model)

        copy_weights(reload_model, path)

        compare_predictions(original_model, reload_model)

    def test_lstm_grown_with_zeroes_predicts_same(self):
        path = os.getcwd() + '/.pytest_cache/test_lstm_grown_with_zeroes_predicts_same.h5'

        original_model = create_model(1, input_shape=(1, 5), middle_layer=LSTM(10, kernel_initializer='random_uniform'))
        original_model.save(path)

        reload_model = create_model(1, input_shape=(1, 5), middle_layer=LSTM(20, kernel_initializer='random_uniform'))
        set_weights_to_zero(reload_model)

        copy_weights(reload_model, path)

        compare_predictions(original_model, reload_model)
