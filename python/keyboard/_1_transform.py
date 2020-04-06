from python.model_training import InMemoryDataSource
import pandas as pd
from python.keyboard._0_import import raw_data, keyboard_layouts  # noqa
from collections import namedtuple


Input = namedtuple('Pair', 'word swipe')
xType = namedtuple('xType', 'a b c')

df = pd.DataFrame(data=[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]], columns=['X', 'y'])
data = InMemoryDataSource(df, 'y')


def encode(word, swipe) -> xType:
    """Converts the specified word and swipe data into a list of features."""
    return [0, 1, 2, 3, 4]


def decode(x: xType) -> Input:
    """Converts the specified list of features into a word and swipe."""
    return Input('', '')
