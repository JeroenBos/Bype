from abc import abstractmethod
from sklearn.base import BaseEstimator
import datetime
from typing import List, Optional, Dict, Any
from typing import TypeVar
from DataSource import DataSource
from tensorflow.keras import Model  # noqa
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping  # noqa
from generic import generic
import numpy as np


class MyBaseEstimator(BaseEstimator):
    models: Dict[str, Model]
    history: Dict[str, List[Any]]
    verbose: bool
    _log_dir: str

    def __init__(self):
        # you cannot add any args here like log_dir: str = 'logs', verbose=False
        # because they will be considered to be hyperparameters by sklearn
        # just set the after having created this object
        super().__init__()
        self.models = {}
        self.history = {}
        self.verbose = False
        self._log_dir = 'logs/'

    def fit_data_source(self, source: DataSource):
        self.fit(source.get_train(), source.get_target())

    # gets called by sklearn, without y
    def fit(self, X, y=None):
        assert hasattr(self, 'num_epochs'), """num_epochs must be present. Set self.num_epochs in __init__"""
        old_X = X  # noqa

        X: np.ndarray = self._preprocess(X)
        model = self.current_model
        self._compile(model)

        log_dir = self._log_dir + datetime.datetime.now().strftime("%Y%m%d-%H")
        callbacks = [
            EarlyStopping(monitor='loss', patience=5),
            TensorBoard(log_dir=log_dir, histogram_freq=1),
            ModelCheckpoint(log_dir + '/model.h5', save_best_only=True, save_weights_only=False)
        ]

        params_repr = self._get_params_repr()
        result = model.fit(X, y, epochs=self.num_epochs, callbacks=callbacks)
        self.history.setdefault(params_repr, []).append(result)
        return result

    def predict(self, X):
        preprocessedX = self._preprocess(X)
        return self._predict(preprocessedX)

    def _predict(self, preprocessedX):
        # if this throws 'Tensor' object has no attribute '_numpy'
        # that could be due to calling _predict before any trainings has occurred
        # eschew calling this method on the initial call to the loss function
        # this initial call is during model compilation and bears a zero-length batch
        return self.current_model.predict(preprocessedX)

    def _preprocess(self, X):
        return X

    @abstractmethod
    def _create_model(self) -> Model:
        pass

    def _compile(self, model: Optional[Model] = None) -> None:
        model = model if model else self.current_model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    @property
    def current_model(self) -> Model:
        params = self._get_params_repr()
        if params not in self.models:
            model = self._create_model()
            assert model is not None, '_create_model() returned None'
            self.models[params] = model
        return self.models[params]

    @property
    def params(self) -> dict:
        result = {**self.__dict__}
        for key in ['models', 'verbose', 'history', '_log_dir', 'preprocessor']:
            if key in result:
                del result[key]
        return result

    def _get_params_repr(self):
        params = sorted(list(self.params.items()), key=lambda t: t[0])
        for entry in params:
            key, value = entry

        def repr(obj) -> str:
            if isinstance(obj, str):
                return "'" + obj + "'"
            return str(obj)

        result = f"({', '.join(f'{entry[0]}={repr(entry[1])}' for entry in params )})"
        return result
