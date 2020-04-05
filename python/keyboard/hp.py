from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import datetime
from typing import List, Optional, Callable, Dict
import tensorflow as tf
from typing import TypeVar
import copy
import python.model_training as mt
from python.model_training import ResultWriter, DataSource
import pandas as pd
from python.keyboard.generic import generic

Model = TypeVar('tensorflow.keras.Models')  # can't find it


class Params(ABC):
    def __init__(self,
                 num_epochs=5,
                 activation='relu'):
        self.num_epochs = num_epochs
        self.activation = activation

    def __str__(self):
        return str(self.__dict__)

    def clone(self) -> "Params":
        return copy.copy(self)

    def getId(self) -> str:
        return str(self)

    def getParameters(self) -> Dict:
        result = {**self.__dict__}
        # not a clue why these shouldn't be removed:
        # del result['getId']
        # del result['getParameters']
        # del result['clone']
        return result


class MLModel(ABC):
    _params: Params
    _model: Optional[Model]

    # A list of fit results of the same parameters
    history: List
    verbose: bool
    _log_dir: str

    def __init__(self, params: Params, log_dir: str = 'logs', verbose=False):
        self._params = params.clone()
        self._model = None

        self.history = None
        self.verbose = verbose
        self._log_dir = log_dir

    def fit(self, X, y):
        if self._model is None:
            self.model = self._create_model()
            self._compile()

        log_dir = self._log_dir + datetime.datetime.now().strftime("%Y%m%d-%H")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        result = self._model.fit(X, y, epochs=self.params.num_epochs, callbacks=[tensorboard_callback])
        self.history.append(result)
        return result

    def predict(self, X):
        X = self._preprocess(X)
        return self._model.predict(X)

    def _preprocess(self, X):
        return X

    @abstractmethod
    def _create_model(self) -> Model:
        pass

    def _compile(self) -> None:
        self._model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

    @property
    def params(self) -> Params:
        return self._params


class AbstractHpEstimator(BaseEstimator, metaclass=generic('TParams')):
    """
    NOTE: HAS UNUSUAL METACLASS:
    example usage:
    AbstractHpEstimator[P](...__init__ args ...)
    where P is a type with Params as supertype

    A class that plays the role of sklearn.Estimator for many models (each characterized by its own parameters)"""

    # a dict from params.getId to the model representing those params
    models = {}
    currentParams: Optional[Params] = None

    _createModel: Callable[[Params], MLModel]

    def __init__(self,
                 modelFactory: Callable[[Params], MLModel]):
        assert 'TParams' not in self.__dict__, "Forgot to specify TParams as type parameter, e.g. HpEstimator[P](...)"
        self._createModel = modelFactory

    # this method is called by sklearn
    def get_params(self, **args):
        return self.params

    # this method is called by sklearn
    def set_params(self, params: Dict) -> None:
        self.currentParams = self.TParams(**params)
        paramsId = self.currentParams.getId()
        if paramsId not in self.models:
            self.models[paramsId] = self._createModel(self.currentParams)

    def get_current_model(self):
        return self.models[self.currentParams.getId()]

    def predict(self, X):
        return self.get_current_model().predict(X)

    def fit(self, X, y):
        return self.get_current_model().fit(X, y)

    # # this method is called by sklearn
    # @classmethod
    # def _get_param_names(cls):
    #     super()._get_param_names(cls.TParam)


def do_hp_search(estimator: AbstractHpEstimator,
                 data_source: DataSource,
                 result_writer: ResultWriter,
                 parameterRanges: Params,
                 combination_number: Optional[int] = None) -> pd.DataFrame:
    hp_searcher = mt.HyperParameterSearcher(scoring='f1',
                                            data_source=data_source,
                                            result_writer=result_writer)

    parameters = parameterRanges.getParameters()
    for key, value in parameters.items():
        if not isinstance(value, List):
            parameters[key] = [value]

    return hp_searcher.fit(estimator_name='Keras',
                           estimator=estimator,
                           parameters=parameters,
                           combination_number=combination_number)
