# this file generates training data
import numpy as np
from keyboard._0_types import T, SwipeDataFrame, SwipeEmbeddingDataFrame, RawTouchEvent, Keyboard
from keyboard._2_transform import keyboards, Key
import pandas as pd
from pandas import DataFrame
from DataSource import InMemoryDataSource, TrivialDataSource
from typing import Callable, List, TypeVar, Any, Union, Tuple
import random
from time import time
from utilities import interpolate, memoize, print_name, windowed_2
import string

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


_single_letters = list(sorted(set(_get_random_str(1) for _ in range(2))))
_double_letters = list(sorted(set(_get_random_str(2) for _ in range(2))))
_triple_letters = list(sorted(set(_get_random_str(3) for _ in range(10))))


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


def generate_random_word(n_chars) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n_chars))

def generate_perfect_lines(n_words: int, n_chars: Union[int, Callable[[int], int]], keyboard_index: int) -> SwipeEmbeddingDataFrame:
    get_n_chars = (lambda i: n_chars) if isinstance(n_chars, int) else n_chars
    words = [generate_random_word(get_n_chars(i)) for i in range(n_words)]
    return SwipeEmbeddingDataFrame.create(words, lambda s, i: generate_perfect_line(s, keyboard_index))


def to_coordinates(word: str, keyboard_or_index: Union[int, Keyboard]) -> List[Tuple[float, float]]:
    keyboard = keyboards[keyboard_or_index] if isinstance(keyboard_or_index, int) else keyboard_or_index

    coords = [(key.abs_center_x, key.abs_center_y) for key in (keyboard.get_key(c) for c in word)]
    return coords


_n_points_per_char = 10, 2  # 10 moving, 2 holding
_dt_per_char = 150
_fractions = [i / (_n_points_per_char[0] - 1) for i in range(_n_points_per_char[0])] + ([1] * _n_points_per_char[1])

def generate_perfect_line(word: str, keyboard_index: int) -> SwipeDataFrame:
    coords = to_coordinates(word, keyboard_index)
    if len(coords) == 1:
        coords *= 2
    path_coords = [interpolate(a, b, f) for a, b in windowed_2(coords) for f in _fractions]

    def get_event(p, i):
        return {
            'PointerIndex': 0,
            # 'Action': int,
            'Timestamp': i * _dt_per_char,
            'X': p[0],
            'Y': p[1],
            # 'Pressure': float,
            # 'Size': float,
            # 'Orientation': float,
            # 'ToolMajor': float,
            # 'ToolMinor': float,
            # 'TouchMinor': float,
            # 'TouchMajor': float,
            # 'XPrecision': float,
            # 'YPrecision': float,
            # 'EdgeFlags': float,
            'KeyboardLayout': keyboard_index,
        }
    return SwipeDataFrame.create(path_coords, get_event)


df = generate_perfect_lines(1, 2, keyboard_index=0)
print(df)

verify = True
generated_data = SwipeEmbeddingDataFrame.__as__(triple_letter_swipes(), verify=verify) 
generated_convolved_data = generated_data.convolve(fraction=1, verify=verify)
