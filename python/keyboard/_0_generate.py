# this file generates training data

from python.keyboard._1_import import SPEC, SPECs  # noqa
from python.keyboard._2_transform import create_empty_swipe_df, keyboards, Key, create_empty_swipe_embedding_df, create_swipe_embedding_df  # noqa
import pandas as pd
from pandas import DataFrame
from python.model_training import InMemoryDataSource, TrivialDataSource  # noqa


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
