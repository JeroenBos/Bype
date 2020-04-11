import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from python.keyboard._0_types import RawTouchEvent

raw_data: pd.DataFrame = pd.read_csv('/home/jeroen/git/bype/data/2020-03-20_0.csv',
                                     names=RawTouchEvent.SPEC.keys(),
                                     dtype=RawTouchEvent.SPEC)


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
