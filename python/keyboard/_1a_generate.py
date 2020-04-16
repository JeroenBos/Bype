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
from GeneratorWithLength import GeneratorWithLength

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



print(time())
_single_letters = GeneratorWithLength(lambda i: _get_random_str(1), 25)
_double_letters = GeneratorWithLength(lambda i: _get_random_str(2), 25)
_triple_letters = GeneratorWithLength(lambda i: _get_random_str(3), 25)


# assert len(list(_single_letters)) == 25
# assert len(list(_triple_letters)) == 25
assert len(list(_single_letters + _triple_letters)) == 50
single_letter_swipes = SwipeEmbeddingDataFrame.create(_single_letters, generate_taps_for)
double_letter_swipes = SwipeEmbeddingDataFrame.create(_double_letters, generate_taps_for)
triple_letter_swipes = SwipeEmbeddingDataFrame.create(_triple_letters, generate_taps_for)

single_and_double_letter_swipes = SwipeEmbeddingDataFrame.create(_single_letters + _double_letters, generate_taps_for)
single_double_and_triple_letter_swipes = SwipeEmbeddingDataFrame.create(_single_letters + _double_letters, generate_taps_for)

print(time())
