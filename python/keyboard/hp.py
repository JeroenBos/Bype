from abc import abstractmethod
from sklearn.base import BaseEstimator
import datetime
from typing import List, Optional, Dict, Any  # noqa
import tensorflow as tf
from typing import TypeVar
import python.model_training as mt
from python.model_training import ResultWriter, DataSource
import pandas as pd
from python.keyboard.generic import generic  # noqa

Model = TypeVar('tensorflow.keras.Models')  # can't find it


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

    # gets called by sklearn
    def fit(self, X, y):
        assert hasattr(self, 'num_epochs'), 'num_epochs must be a parameter on this estimator. Set it in your __init__ to self'
        old_X = X  # noqa

        X = self._preprocess(X)
        model = self.current_model
        self._compile(model)

        log_dir = self._log_dir + datetime.datetime.now().strftime("%Y%m%d-%H")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        params_repr = self._get_params_repr()
        result = model.fit(X.to_numpy(), y.to_numpy(), epochs=self.num_epochs, callbacks=[tensorboard_callback])
        self.history.setdefault(params_repr, []).append(result)
        return result

    def predict(self, X):
        X = self._preprocess(X)
        return self.current_model.predict(X)

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
        for key in ['models', 'verbose', 'history', '_log_dir']:
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


def do_hp_search(TEstimator: type,
                 data_source: DataSource,
                 result_writer: ResultWriter,
                 parameterRanges: dict,
                 combination_number: Optional[int] = None) -> pd.DataFrame:
    hp_searcher = mt.HyperParameterSearcher(scoring='f1',
                                            data_source=data_source,
                                            result_writer=result_writer)

    initial_params = {}

    for key, value in parameterRanges.items():
        if not isinstance(value, List):
            parameterRanges[key] = [value]
        initial_params[key] = parameterRanges[key][0]

    estimator = TEstimator(**initial_params)

    return hp_searcher.fit(estimator_name='Keras',
                           estimator=estimator,
                           parameters=parameterRanges,
                           combination_number=combination_number)
