from python.model_training import InMemoryDataSource
import pandas as pd
from python.keyboard._1_import import raw_data, keyboard_layouts, KEYBOARD_LAYOUT_SPEC  # noqa
from collections import namedtuple
from typing import Dict, List, Union  # noqa


Input = namedtuple('Pair', 'word swipe')
xType = namedtuple('xType', 'a b c')

df = pd.DataFrame(data=[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]], columns=['X', 'y'])
data = InMemoryDataSource(df, 'y')


def encode(swipe: pd.Series, word: pd.Series) -> xType:
    """Converts the specified word and swipe data into a list of features."""
    return [0, 1, 2, 3, 4]


def decode(x: xType) -> Input:
    """Converts the specified list of features into a word and swipe."""
    return Input('', '')


class Key:
    def __init__(self, code: int, x: int, y: int, width: int, height: int,
                 edgeFlags: int, repeatable: bool, toggleable: bool):
        self.code = code
        self.x = x
        self.y = y
        self.width = width
        self.edge_flags = edgeFlags
        self.repeatable = repeatable
        self.toggleable = toggleable


def get_keyboard(keyboard_layout: Union[int, pd.DataFrame]) -> Dict[int, Key]:
    result = List[Key]
    keyboard = keyboard_layouts[keyboard_layout] if isinstance(keyboard_layout, int) else keyboard_layout

    def key_from_row(row, code):
        return Key(code=code, **{f'{col}': row[col] for col in keyboard.columns.values if col != 'codes'})

    allcodes = list(code for _index, row in keyboard.iterrows() for code in row['codes'])
    if len(allcodes) != len(set(allcodes)):
        raise KeyError("The same code at multiple places isn't supported yet")

    result: Dict[int, Key] = {code: key_from_row(row, code)
                              for _index, row in keyboard.iterrows()
                              for code in row['codes']}
    return result


