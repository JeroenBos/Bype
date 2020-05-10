import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Union, Dict
from keyboard._0_types import RawTouchEvent, Key, Keyboard, SwipeDataFrame, RawTouchEventActions, SwipeEmbeddingDataFrame
from utilities import get_resource, incremental_paths, memoize, windowed_2, read_json, concat, is_list_of, read_all, skip, split_at, split_by

@memoize
def read_raw_data(path: str):
    return pd.read_csv(path, names=RawTouchEvent.SPEC.keys(), dtype=RawTouchEvent.SPEC)

def split_on_down_action(df: SwipeDataFrame) -> List[SwipeDataFrame]:
    """ Splits a swipe dataframe into a dataframe per swipe. """
    existing_actions = list(sorted(set(df.Action)))
    if existing_actions != [RawTouchEventActions.Down, RawTouchEventActions.Up, RawTouchEventActions.Move]:
        raise ValueError('Unsupported action deteted')

    start_indices = df.index[df.Action == RawTouchEventActions.Down].tolist()
    return [df.loc[a:b, :] for a, b in windowed_2(start_indices, 0, len(df))]

def text_file_to_words(path: str) -> List[str]:
    raw_data_text = read_all(path)
    return text_to_words(raw_data_text)

def text_to_words(raw_data_text: str) -> List[str]:

    def split_punctuation(s: str):
        assert isinstance(s, str)
        s = s.replace(' ', '').replace('\r', '')
        split_indices = concat((i, i + 1) for i, c in enumerate(s) if not c.isalpha())

        return split_at(s, *split_indices)

    def is_valid(s: str):
        return len(s) != 0 and not s.isspace()

    words = split_by(raw_data_text, '\n', ' ')  # word here means more generic than string of characters: it's more of a string of values (values represented on keys)
    words = concat(split_punctuation(word) for word in words)
    words = [word for word in words if is_valid(word)]
    return words


def correct(words, swipes, frames_to_skip: List[int], extra_swipes: List[int], words_to_merge: List[List[int]], empty_swipe=None):
    for combi in words_to_merge:
        words[combi[0]] = "".join(words[c] for c in combi)
        for r in sorted(combi[1:], reverse=True):
            del words[r]

    words_iter = iter(words)
    word_index = -1

    def next_word():
        nonlocal word_index
        word_index += 1
        try:
            return str(word_index) + ': ' + next(words_iter)
        except StopIteration:
            return str(word_index) + ': <no word>'


    for i, frame in enumerate(swipes):
        if i in frames_to_skip:
            continue
        if i in extra_swipes:
            yield empty_swipe, next_word()

        yield frame, next_word()

def correct_and_skip(skip_count, *args):
    return skip(correct(*args), skip_count)


def wrap(file_name, swipes_to_skip, extra_frames, words_to_merge):
    raw_data = split_on_down_action(read_raw_data(get_resource(file_name + '.csv')))
    raw_data_text = text_file_to_words(get_resource(file_name + '.txt'))

    swipes, words = tuple(zip(*list(correct(raw_data_text, raw_data, swipes_to_skip, extra_frames, words_to_merge))))

    return SwipeEmbeddingDataFrame.create(words, lambda word, i: swipes[i], verify=True), swipes, words

@memoize
def _2020_03_20_0():
    return wrap('2020-03-20_0', [0, 1, 2, 3], [], [[11, 12, 13]])


_2020_03_20_0_computed = _2020_03_20_0()

# ########### KEYBOARDS ############


KEY_SPEC = {
    "codes": np.int32,  # this is technically not correct because it's an array of int32s. But pd accepts it... ???
    "x": np.int32,
    "y": np.int32,
    "width": np.int32,
    "height": np.int32,
    "edgeFlags": np.int32,
    "repeatable": np.bool,
    "toggleable": np.bool,
}

# KEYBOARD_LAYOUT_SPEC = {
#     "width": np.int32,
#     "height": np.int32,
#     "keys": List[KEY_SPEC],
# }

def _loadLayoutFile(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(get_resource('empty.csv'),
                                   names=KEY_SPEC.keys(),
                                   dtype=KEY_SPEC)
    assert len(df.columns) == len(KEY_SPEC)
    keyboard_json = read_json(path)
    layout = pd.read_json(json.dumps(keyboard_json['keys']))
    df = df.append(layout)
    assert len(df.columns) == len(KEY_SPEC)

    setattr(df, 'keyboard_left', keyboard_json['left'])
    setattr(df, 'keyboard_top', keyboard_json['top'])
    setattr(df, 'keyboard_width', keyboard_json['width'])
    setattr(df, 'keyboard_height', keyboard_json['height'])
    return df



def get_keyboard(keyboard_layout: Union[int, pd.DataFrame]) -> Dict[int, Key]:
    result = List[Key]
    keyboard = _keyboard_layouts[keyboard_layout] if isinstance(keyboard_layout, int) else keyboard_layout

    def key_from_row(row, code, code_index):
        return Key(code, code_index, **{f'{col}': row[col] for col in keyboard.columns.values if col != 'codes'})

    result: Dict[int, Key] = {code: key_from_row(row, code, code_index)
                              for _index, row in keyboard.iterrows()
                              for code_index, code in enumerate(row['codes'])}

    layout_id = keyboard_layout if isinstance(keyboard_layout, int) else '?'

    return Keyboard(layout_id, 
                    keyboard.keyboard_width,
                    keyboard.keyboard_height,
                    keyboard.keyboard_left,
                    keyboard.keyboard_top,
                    iterable=result)


# loads all json layouts of the specified format
_json_layout_format = get_resource('keyboardlayout_%d.json')
_keyboard_layouts: List[pd.DataFrame] = [_loadLayoutFile(path) for path in incremental_paths(_json_layout_format)]

# HACK:
_keyboard_layouts[0].keyboard_top = 100  # TODO: find out why this offset exists relative to motion data

# interpret all json layouts
keyboards: List[Keyboard] = [get_keyboard(keyboard_index) for keyboard_index in range(len(_keyboard_layouts))]  # by index for ids
