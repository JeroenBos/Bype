from python.model_training import InMemoryDataSource
import pandas as pd
from python.keyboard._0_types import Key, Keyboard, SwipeDataFrame, Input
from python.keyboard._1_import import raw_data, keyboard_layouts, KEYBOARD_LAYOUT_SPEC, SPEC, SPECs
from collections import namedtuple
from typing import Dict, List, Union, TypeVar, Callable, Tuple


xType = namedtuple('xType', 'a b c')

df = pd.DataFrame(data=[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]], columns=['X', 'y'])
data = InMemoryDataSource(df, 'y')


def get_code(char: str) -> int:
    assert isinstance(char, str)
    assert len(char) == 1

    return ord(char[0])


def _get_keyboard(touchevent: pd.Series) -> Keyboard:
    assert isinstance(SPECs.keyboard_layout, str), 'WTF. The property literally returns a string...'
    assert hasattr(touchevent, SPECs.keyboard_layout)
    layout_id = touchevent[SPECs.keyboard_layout]
    assert isinstance(layout_id, int)
    keyboard = keyboards[layout_id]
    return keyboard


def _get_normalized_x(touchevent: pd.Series) -> float:
    return _get_keyboard(touchevent).normalize_x(touchevent[SPECs.x])


def _get_normalized_y(touchevent: pd.Series) -> float:
    return _get_keyboard(touchevent).normalize_y(touchevent[SPECs.y])


def _get_key(char: str, touchevent: pd.Series) -> Key:
    code = get_code(char)
    keyboard = _get_keyboard(touchevent)
    if code not in keyboard:
        return Key.NO_KEY
    else:
        key = keyboard[code]
        assert isinstance(key, Key), "Maybe later a list of keys can be implemented?"
        return key


def _get_normalized_button_x(word: str, index: int) -> Callable[[pd.Series], float]:
    def impl(touchevent: pd.Series) -> float:
        if index >= len(word):
            return -1

        key: Key = _get_key(word[index], touchevent)
        return key.keyboard.normalize_x(key.x + key.width / 2)
    return impl


def _get_normalized_button_y(word: str, index: int) -> Callable[[pd.Series], float]:
    def impl(touchevent: pd.Series) -> float:
        if index >= len(word):
            return -1

        key: Key = _get_key(word[index], touchevent)
        return key.keyboard.normalize_y(key.y + key.height / 2)
    return impl


def encode(swipe: SwipeDataFrame, word: str) -> xType:
    """
    Converts the specified word and swipe data into a list of features.
    :param swipe: A row in DataSource.get_train().
    :param word: A row in DataSource.get_target().
    """
    assert SwipeDataFrame.is_instance(swipe)
    assert isinstance(word, str)
    assert 'X' in swipe
    assert 'Y' in swipe


    features_per_time_step: List[Callable[[pd.Series], float]] = [
        _get_normalized_x,
        _get_normalized_y,
        _get_normalized_button_x(word, 0),
        _get_normalized_button_y(word, 0),
        _get_normalized_button_x(word, 1),
        _get_normalized_button_y(word, 1),
        _get_normalized_button_x(word, 2),
        _get_normalized_button_y(word, 2),
    ]

    result: List[List[float]] = []
    for i, touchevent in swipe.iterrows():
        time_step = []
        for feature_per_time_step in features_per_time_step:
            time_step.append(feature_per_time_step(touchevent))
        result.append(time_step)
    return result


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

    return Keyboard(layout_id, width, height, left, top, iterable=result)


keyboards: List[Keyboard] = [get_keyboard(layout_index) for layout_index in range(len(keyboard_layouts))]
