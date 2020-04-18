# this file generates training data
import numpy as np
from keyboard._0_types import T, SwipeDataFrame, SwipeEmbeddingDataFrame, RawTouchEvent
from keyboard._1_import import KEYBOARD_LAYOUT_SPEC
from keyboard._2_transform import keyboards, Key
import pandas as pd
from pandas import DataFrame
from DataSource import InMemoryDataSource, TrivialDataSource
from typing import Callable, List, TypeVar, Any, Union
import random
from time import time
from utilities import memoize, print_name

def generate_taps_for(word: str, i=None) -> SwipeDataFrame:
    """ Creates a 'swipe' as a sequence of perfect taps. """
    assert isinstance(word, str)
    assert len(word) != 0

    # first generate dataframe of the correct format and size:

    def create_event(char: str, i: int):
        assert isinstance(word, str)

        keyboard = keyboards[0]
        code = ord(char)
        if code not in keyboard:
            raise ValueError(f"Character '{char}' was not found on the keyboard")
        key: Key = keyboard[code]


        event: RawTouchEvent = {
            "X": key.x + key.width / 2,
            "Y": key.y + key.height / 2,
            "KeyboardLayout": keyboard.layout_id,
            "KeyboardWidth": keyboard.width,
            "KeyboardHeight": keyboard.height,
        }

        assert isinstance(keyboard.layout_id, int)

        assert isinstance(event["X"], RawTouchEvent.SPEC["X"]) 
        assert isinstance(event["Y"], RawTouchEvent.SPEC["Y"]) 
        assert isinstance(event["KeyboardLayout"], RawTouchEvent.SPEC["KeyboardLayout"])
        return event

    result = SwipeDataFrame.create(word, create_event)
    return result


_letters = [chr(i) for i in range(97, 97 + 26)]
def _get_random_str(length: int):
    return "".join(random.sample(_letters, length))


_single_letters = list(sorted(set(_get_random_str(1) for _ in range(200))))
_double_letters = list(sorted(set(_get_random_str(2) for _ in range(5))))
_triple_letters = list(sorted(set(_get_random_str(3) for _ in range(5))))


@memoize
@print_name
def single_letter_swipes():
    return SwipeEmbeddingDataFrame.create(_single_letters, generate_taps_for)
@memoize
@print_name
def double_letter_swipes():
    return SwipeEmbeddingDataFrame.create(_double_letters, generate_taps_for)
@memoize
@print_name
def triple_letter_swipes():
    return SwipeEmbeddingDataFrame.create(_triple_letters, generate_taps_for)

@memoize
@print_name
def single_and_double_letter_swipes():
    return SwipeEmbeddingDataFrame.create(_single_letters + _double_letters, generate_taps_for)
@memoize
@print_name
def single_double_and_triple_letter_swipes():
    return SwipeEmbeddingDataFrame.create(_single_letters + _double_letters + _triple_letters, generate_taps_for)
