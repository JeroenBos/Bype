from tensorflow.keras.models import load_model, Model  # noqa
from tensorflow.keras.layers import Layer  # noqa
from typing import Union
import numpy as np


def copy_weights(model_to_init: Model, model_to_copy_from: Union[str, Model]) -> None:
    if isinstance(model_to_copy_from, str):
        model_to_copy_from = load_model(model_to_copy_from)

    if len(model_to_copy_from.layers) != len(model_to_init.layers):
        raise ValueError('Incompatible models')

    for _to, _from in zip(model_to_init.layers, model_to_copy_from.layers):
        _copy_weights(_from, _to)


def _copy_weights(_from: Layer, _to: Layer) -> None:
    if len(_from.weights) == 0:
        assert len(_to.weights) == 0
        return

    old_shape = _from.weights[0].shape
    old_bias_shape = _from.weights[1].shape
    new_shape = _to.weights[0].shape
    new_bias_shape = _to.weights[1].shape

    if len(old_shape) != len(new_shape):
        raise ValueError("The layers have unequal ranks")

    if len(old_bias_shape) != len(new_bias_shape):
        raise ValueError("The layers have unequal bias ranks")

    copy_shape = tuple(min(old, new) for old, new in zip(old_shape, new_shape))
    copy_bias_shape = tuple(min(old, new) for old, new in zip(old_bias_shape, new_bias_shape))

    new_weights = _copy(_from.weights[0], _to.weights[0].numpy(), copy_shape)
    new_bias_Ws = _copy(_from.weights[1], _to.weights[1].numpy(), copy_bias_shape)

    _to.weights[0].assign(new_weights)
    _to.weights[1].assign(new_bias_Ws)



def _copy(_from: np.array, _to: np.array, shape: tuple) -> np.array:
    assert len(_from.shape) == len(_to.shape) == len(shape)

    print(type(_to))
    if len(shape) == 1:
        _to[:shape[0]] = _from[:shape[0]]
    elif len(shape) == 2:
        _to[:shape[0], :shape[1]] = _from[:shape[0], :shape[1]]
    elif len(shape) == 3:
        _to[:shape[0], :shape[1], :shape[2]] = _from[:shape[0], :shape[1], :shape[2]]
    elif len(shape) == 4:
        _to[:shape[0], :shape[1], :shape[2], :shape[3]] = _from[:shape[0], :shape[1], :shape[2], :shape[3]]
    else:
        raise ValueError('not implemented')
    return _to
