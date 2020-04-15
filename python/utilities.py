import pandas as pd

def print_fully(df: pd.DataFrame) -> None:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
