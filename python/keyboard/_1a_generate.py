# this file generates training data
import numpy as np
from python.keyboard._0_types import T, SwipeDataFrame, SwipeEmbeddingDataFrame, RawTouchEvent
from python.keyboard._1_import import KEYBOARD_LAYOUT_SPEC
from python.keyboard._2_transform import keyboards, Key
import pandas as pd
from pandas import DataFrame
from python.model_training import InMemoryDataSource, TrivialDataSource
from typing import Callable, List, TypeVar, Any, Union



def generate_taps_for(word: str) -> SwipeDataFrame:
    """ Creates a 'swipe' as a sequence of perfect taps. """

    # first generate dataframe of the correct format and size:
    result = SwipeDataFrame.create_empty(1)

    keyboard = keyboards[0]
    char = ord(word[0])
    if char not in keyboard:
        raise ValueError(f"Character '{word[0]}' was not found on the keyboard")
    key: Key = keyboard[char]

    assert isinstance(keyboard.layout_id, int)
    result.X[0] = key.x + key.width / 2
    result.Y[0] = key.y + key.height / 2
    result.KeyboardLayout[0] = keyboard.layout_id
    result.KeyboardWidth[0] = keyboard.width
    result.KeyboardHeight[0] = keyboard.height

    assert isinstance(result.X[0], np.int64) 
    assert isinstance(result.Y[0], np.int64) 
    assert isinstance(result.KeyboardLayout[0], np.int64)

    return result


_single_letters = [chr(i) for i in range(97, 97 + 26)]
_single_letter_words = DataFrame(_single_letters, columns=['word'], dtype=str)

single_letter_swipes = SwipeEmbeddingDataFrame.create(_single_letters, lambda word, i: generate_taps_for(word))
