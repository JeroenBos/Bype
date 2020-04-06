import pandas
import numpy as np
from pathlib import Path


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

raw_data: pandas.DataFrame = pandas.read_csv('/home/jeroen/git/bype/data/2020-03-20_0.csv',
                                             names=SPEC.keys(),
                                             dtype=SPEC)


KEYBOARD_LAYOUT_SPEC = {
    "codes": np.int32,  # this is technically not correct because it's an array of int32s. But pandas accepts it... ???
    "x": np.int32,
    "y": np.int32,
    "width": np.int32,
    "height": np.int32,
    "edgeFlags": np.int32,
    "repeatable": np.bool,
    "toggleable": np.bool,
}


keyboard_layouts = pandas.DataFrame()
for column in KEYBOARD_LAYOUT_SPEC:
    keyboard_layouts[column] = pandas.Series(dtype=KEYBOARD_LAYOUT_SPEC[column])


def _loadLayoutFile(path: str) -> None:
    layout = pandas.read_json(path)
    globals()['keyboard_layouts'] = layout


# Load all keyboard layout files
i = 0
while True:
    path = f'/home/jeroen/git/bype/data/keyboardlayout_{i}.json'
    if(not Path(path).exists()):
        break
    _loadLayoutFile(path)
    i = i + 1
