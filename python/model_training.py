from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    def get_train(self):
        raise NotImplementedError

    @abstractmethod
    def get_target(self):
        raise NotImplementedError
