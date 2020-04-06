from python.model_training import InMemoryDataSource
import pandas as pd
from python.keyboard._0_import import raw_data, keyboard_layouts  # noqa


df = pd.DataFrame(data=[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]], columns=['X', 'y'])
data = InMemoryDataSource(df, 'y')
