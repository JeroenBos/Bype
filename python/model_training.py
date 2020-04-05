from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Union


class DataSource(ABC):
    @abstractmethod
    def get_train(self):
        raise NotImplementedError

    @abstractmethod
    def get_target(self):
        raise NotImplementedError


class CsvDataSource(DataSource):
    def __init__(self, path: str, target_column_name_or_index: Union[int, str], *read_csv_args):
        self.path = path
        self.df = pd.read_csv(path, *read_csv_args)
        x = target_column_name_or_index  # shorten name
        if isinstance(x, str):
            self.target_column_name = x
        else:
            self.target_column_name = list(self.df.columns.values)[x]
            assert self.target_column_name is not None

    def get_target(self):
        return self.df[:, self.target_column_index]

    def get_train(self):
        return self.df.drop([self.target_column])


class InMemoryDataSource(DataSource):
    def __init__(self, path: str, target_column_name_or_index: Union[int, str], *read_csv_args):
        self.path = path
        self.df = pd.read_csv(path, *read_csv_args)
        x = target_column_name_or_index  # shorten name
        if isinstance(x, str):
            self.target_column_name = x
        else:
            self.target_column_name = list(self.df.columns.values)[x]
            assert self.target_column_name is not None

    def get_target(self):
        return self.df[:, self.target_column_index]

    def get_train(self):
        return self.df.drop([self.target_column])
