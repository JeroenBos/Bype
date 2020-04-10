# this file generates training data

from python.keyboard._1_import import SPEC, SPECs  # noqa
from python.keyboard._2_transform import keyboards, Key  # noqa
import pandas as pd
from pandas import DataFrame
from python.model_training import InMemoryDataSource, TrivialDataSource  # noqa
from typing import Callable, List, TypeVar

T = TypeVar('T')


def create_empty_swipe_df(length: int, **defaults) -> pd.DataFrame:
    """
    Creates an empty df of the correct format and shape determined by SPEC,
    initialized with default values, which default to 0.
    """
    return create_empty_df(length, columns=list(SPEC.keys()), **defaults)


def create_empty_swipe_embedding_df(length: int) -> pd.DataFrame:
    defaults = {'swipes': create_empty_swipe_df(0)}
    return create_empty_df(length, columns=['swipes'], **defaults)


def create_swipe_embedding_df(inputs: List[T], swipe_selector: Callable[[T], pd.DataFrame]) -> pd.DataFrame:
    result = create_empty_swipe_embedding_df(len(inputs))
    swipes = [swipe_selector(input) for input in inputs]

    for swipe in swipes:
        assert isinstance(swipe, pd.DataFrame)
        assert len(swipe) != 0, 'empty swipe'
    assert len(set(len(swipe.columns) for swipe in swipes)) == 1, 'not all swipes have same columns'

    for i, swipe in enumerate(swipes):
        result['swipes'][i] = swipe
    return result


def create_empty_df(length: int, columns: List[str], **defaults) -> pd.DataFrame:
    """ Creates an empty df of the correct format and shape, initialized with default values, which default to 0. """
    for key, value in defaults.items():
        if key not in set(columns):
            raise ValueError(f"Unexpected default value specified for '{str(key)}'")

    for column in columns:
        if column not in defaults:
            defaults[column] = 0

    result = pd.DataFrame([list(defaults[key] for key in columns) for _ in range(length)], columns=columns)
    return result


def generate_taps_for(word: str) -> pd.DataFrame:
    """ Creates a 'swipe' as a sequence of perfect taps. """

    # first generate dataframe of the correct format and size:
    result = create_empty_swipe_df(1)

    keyboard = keyboards[0]
    char = ord(word[0])
    if char not in keyboard:
        raise ValueError(f"Character '{word[0]}' was not found on the keyboard")
    key: Key = keyboard[char]

    result[SPECs.x] = key.x
    result[SPECs.y] = key.y
    result[SPECs.keyboard_layout] = keyboard.layout_id
    result[SPECs.keyboard_width] = keyboard.width
    result[SPECs.keyboard_height] = keyboard.height

    return result


_single_letters = [chr(i) for i in range(97, 97 + 26)]
_single_letter_words = DataFrame(_single_letters, columns=['word'], dtype=str)

single_letter_swipes = create_swipe_embedding_df(_single_letters, generate_taps_for)
