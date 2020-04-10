from python.model_training import InMemoryDataSource
import pandas as pd
from python.keyboard._0_types import Key, Keyboard
from python.keyboard._1_import import raw_data, keyboard_layouts, KEYBOARD_LAYOUT_SPEC, SPEC  # noqa
from collections import namedtuple
from typing import Dict, List, Union, TypeVar, Callable  # noqa


Input = namedtuple('Pair', 'word swipe')
xType = namedtuple('xType', 'a b c')

df = pd.DataFrame(data=[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]], columns=['X', 'y'])
data = InMemoryDataSource(df, 'y')


def encode(swipe: pd.Series, word: pd.Series) -> xType:
    """
    Converts the specified word and swipe data into a list of features.
    :param swipe: A row in DataSource.get_train().
    :param word: A row in DataSource.get_target().
    """
    assert isinstance(swipe, pd.Series)
    assert isinstance(word, pd.Series)
    assert 'X' in swipe
    assert 'Y' in swipe
    assert 'word' in word
    return [0, 1, 2, 3, 4]


def decode(x: xType) -> Input:
    """Converts the specified list of features into a word and swipe."""
    return Input('', '')


def get_keyboard(keyboard_layout: Union[int, pd.DataFrame]) -> Dict[int, Key]:
    result = List[Key]
    keyboard = keyboard_layouts[keyboard_layout] if isinstance(keyboard_layout, int) else keyboard_layout

    def key_from_row(row, code, code_index):
        return Key(code, code_index, **{f'{col}': row[col] for col in keyboard.columns.values if col != 'codes'})

    # allcodes = list(code for _index, row in keyboard.iterrows() for code in row['codes'])
    # if len(allcodes) != len(set(allcodes)):
    #     raise KeyError("The same code at multiple places isn't supported yet")

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

    return Keyboard(layout_id, width, height, iterable=result)


keyboards: List[Keyboard] = [get_keyboard(layout_index) for layout_index in range(len(keyboard_layouts))]
