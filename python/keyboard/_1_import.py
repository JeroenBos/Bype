import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Union, Dict
from keyboard._0_types import RawTouchEvent, Key, Keyboard, SwipeDataFrame, RawTouchEventActions
from utilities import get_resource, incremental_paths, memoize, windowed_2, read_json

@memoize
def read_raw_data(path: str):
    return pd.read_csv(path, names=RawTouchEvent.SPEC.keys(), dtype=RawTouchEvent.SPEC)

def split_words(df: SwipeDataFrame) -> List[SwipeDataFrame]:
    """ Splits a swipe dataframe into a dataframe per swipe. """
    existing_actions = list(sorted(set(df.Action)))
    if existing_actions != [RawTouchEventActions.Down, RawTouchEventActions.Up, RawTouchEventActions.Move]:
        raise ValueError('Unsupported action deteted')

    start_indices = df.index[df.Action == RawTouchEventActions.Down].tolist()
    return [df.loc[a:b, :] for a, b in windowed_2(start_indices, 0, len(df))]


raw_data = split_words(read_raw_data(get_resource('2020-03-20_0.csv')))


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
