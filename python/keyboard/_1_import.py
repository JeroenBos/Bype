import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


SPEC = {
    "PointerIndex": np.int32,
    "Action": np.int32,
    "Timestamp": np.int64,
    "X": np.float32,
    "Y": np.float32,
    "Pressure": np.float32,
    "Size": np.float32,
    "Orientation": np.float32,
    "ToolMajor": np.float32,
    "ToolMinor": np.float32,
    "TouchMinor": np.float32,
    "TouchMajor": np.float32,
    "XPrecision": np.float32,
    "YPrecision": np.float32,
    "EdgeFlags": np.float32,
    "KeyboardLayout": np.int32,
    "KeyboardWidth": np.int32,
    "KeyboardHeight": np.int32,
}


class SPECs:
    @property
    def pointer_index(self) -> float:
        return "PointerIndex"

    @property
    def action(self) -> float:
        return "Action"

    @property
    def time_stap(self) -> float:
        return "Timestamp"

    @property
    def x(self) -> float:
        return "X"

    @property
    def y(self) -> float:
        return "Y"

    @property
    def pressure(self) -> float:
        return "Pressure"

    @property
    def size(self) -> float:
        return "Size"

    @property
    def orientation(self) -> float:
        return "Orientation"

    @property
    def tool_major(self) -> float:
        return "ToolMajor"

    @property
    def tool_minor(self) -> float:
        return "ToolMinor"

    @property
    def touch_minor(self) -> float:
        return "TouchMinor"

    @property
    def touch_major(self) -> float:
        return "TouchMajor"

    @property
    def x_precision(self) -> float:
        return "XPrecision"

    @property
    def y_precision(self) -> float:
        return "YPrecision"

    @property
    def edge_flags(self) -> float:
        return "EdgeFlags"

    @property
    def keyboard_layout(self) -> float:
        return "KeyboardLayout"

    @property
    def keyboard_width(self) -> float:
        return "KeyboardWidth"

    @property
    def keyboard_height(self) -> float:
        return "KeyboardHeight"


raw_data: pd.DataFrame = pd.read_csv('/home/jeroen/git/bype/data/2020-03-20_0.csv',
                                     names=SPEC.keys(),
                                     dtype=SPEC)


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


keyboard_layouts: List[pd.DataFrame] = []


def _loadLayoutFile(path: str) -> None:
    df: pd.DataFrame = pd.read_csv('/home/jeroen/git/bype/data/empty.csv',
                                   names=KEYBOARD_LAYOUT_SPEC.keys(),
                                   dtype=KEYBOARD_LAYOUT_SPEC)
    assert len(df.columns) == len(KEYBOARD_LAYOUT_SPEC)
    layout = pd.read_json(path)
    df = df.append(layout)
    assert len(df.columns) == len(KEYBOARD_LAYOUT_SPEC)

    keyboard_layouts.append(df)


# Load all keyboard layout files
i = 0
while True:
    path = f'/home/jeroen/git/bype/data/keyboardlayout_{i}.json'
    if(not Path(path).exists()):
        break
    _loadLayoutFile(path)
    i = i + 1
