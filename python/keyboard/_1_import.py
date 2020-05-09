import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Union, Dict
from keyboard._0_types import RawTouchEvent, Key, Keyboard
from utilities import incremental_paths

raw_data: pd.DataFrame = pd.read_csv('/home/jeroen/git/bype/data/2020-03-20_0.csv',
                                     names=RawTouchEvent.SPEC.keys(),
                                     dtype=RawTouchEvent.SPEC)



# ########### KEYBOARDS ############


KEYBOARD_LAYOUT_SPEC = {
    "codes": np.int32,  # this is technically not correct because it's an array of int32s. But pd accepts it... ???
    "x": np.int32,
    "y": np.int32,
    "width": np.int32,
    "height": np.int32,
    "edgeFlags": np.int32,
    "repeatable": np.bool,
    "toggleable": np.bool,
}



def _loadLayoutFile(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv('/home/jeroen/git/bype/data/empty.csv',
                                   names=KEYBOARD_LAYOUT_SPEC.keys(),
                                   dtype=KEYBOARD_LAYOUT_SPEC)
    assert len(df.columns) == len(KEYBOARD_LAYOUT_SPEC)
    layout = pd.read_json(path)
    df = df.append(layout)
    assert len(df.columns) == len(KEYBOARD_LAYOUT_SPEC)

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

    # infer keyboard width and height:
    lefts = (key.x for key in result.values())
    rights = (key.x + key.width for key in result.values())
    tops = (key.y for key in result.values())
    bottoms = (key.y + key.height for key in result.values())
    left = min(lefts)
    right = max(rights)
    top = min(tops)
    bottom = max(bottoms)
    width = right - left
    height = bottom - top

    return Keyboard(layout_id, width, height, left, top, iterable=result)


# loads all json layouts of the specified format
_json_layout_format = '/home/jeroen/git/bype/data/keyboardlayout_%d.json'
_keyboard_layouts: List[pd.DataFrame] = [_loadLayoutFile(path) for path in incremental_paths(_json_layout_format)]

# interpret all json layouts
keyboards: List[Keyboard] = [get_keyboard(keyboard_index) for keyboard_index in range(len(_keyboard_layouts))]  # by index for ids
