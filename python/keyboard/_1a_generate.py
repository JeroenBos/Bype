# this file generates training data
import numpy as np
from python.keyboard._0_types import T, SwipeDataFrame, SwipeEmbeddingDataFrame, RawTouchEvent
from python.keyboard._1_import import KEYBOARD_LAYOUT_SPEC
from python.keyboard._2_transform import keyboards, Key
import pandas as pd
from pandas import DataFrame
from python.model_training import InMemoryDataSource, TrivialDataSource
from typing import Callable, List, TypeVar, Any, Union


def create_empty_swipe_df(length: int, **defaults) -> SwipeDataFrame:
    """
    Creates an empty df of the correct format and shape determined by SPEC,
    initialized with default values, which default to 0.
    """
    return create_empty_df(length, columns=RawTouchEvent.get_keys(), **defaults)


def create_empty_swipe_embedding_df(length: int) -> SwipeEmbeddingDataFrame:
    defaults = {'swipes': create_empty_swipe_df(0), 'words': pd.Series([], dtype=np.str)}
    return create_empty_df(length, columns=list(defaults.keys()), **defaults)


def create_swipe_embedding_df(words: List[str], swipe_selector: Callable[[str, int], SwipeDataFrame]) -> SwipeEmbeddingDataFrame:
    result = create_empty_swipe_embedding_df(len(words))
    swipes = [swipe_selector(word, i) for i, word in enumerate(words)]

    for swipe in swipes:
        assert isinstance(swipe, pd.DataFrame)
        assert len(swipe) != 0, 'empty swipe'
    assert len(set(len(swipe.columns) for swipe in swipes)) == 1, 'not all swipes have same columns'

    for i, swipe in enumerate(swipes):
        result.swipes[i] = swipe
        result.words[i] = words[i]
    return result


def create_empty_df(length: int, columns: List[str], **defaults: Union[Any, Callable[[], Any]]) -> pd.DataFrame:
    """ Creates an empty df of the correct format and shape, initialized with default values, which default to 0. """
    for key, value in defaults.items():
        if key not in set(columns):
            raise ValueError(f"Unexpected default value specified for '{str(key)}'")

    # get lazily evaluable defaults
    defaults = {key: (value() if isinstance(value, Callable) else value) for key, value in defaults.items()}

    # pad defaults with zeroes
    for column in columns:
        if column not in defaults:
            defaults[column] = 0

    result = pd.DataFrame([list(defaults[key] for key in columns) for _ in range(length)], columns=columns)
    return result


def generate_taps_for(word: str) -> SwipeDataFrame:
    """ Creates a 'swipe' as a sequence of perfect taps. """

    # first generate dataframe of the correct format and size:
    result = create_empty_swipe_df(1)

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

single_letter_swipes = create_swipe_embedding_df(_single_letters, lambda word, i: generate_taps_for(word))
