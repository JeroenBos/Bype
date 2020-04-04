from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import datetime
from typing import List, Optional, Callable, Dict
import tensorflow as tf
from typing import TypeVar
import copy
import python.model_training as mt
from python.model_training import ResultWriter, DataSource


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

    @abstractmethod
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


class HpEstimator(BaseEstimator):
    """ A class that plays the role of sklearn.Estimator for many models (each characterized by its own parameters)"""

    # a dict from params.getId to the model representing those params
    models = {}
    currentParams: Optional[Params] = None

    TParam: type
    _createModel: Callable[[Params], MLModel]

    def __init__(self,
                 modelFactory: Callable[[Params], MLModel],
                 TParams: type):
        self._createModel = modelFactory
        self.TParams = TParams

    def get_params(self, **args):
        return self.params

    def set_params(self, params: Params) -> None:
        self.currentParams = params.clone() if params is not None else None
        paramsId = params.getId()
        if paramsId not in self.models:
            self.models[paramsId] = self._createModel(params)

    def get_current_model(self):
        return self.models[self.currentParams.getId()]

    def predict(self, X):
        return self.get_current_model().predict(X)

    def fit(self, X, y):
        return self.get_current_model().fit(X, y)


def do_hp_search(estimator: HpEstimator,
                 data_source: DataSource,
                 result_writer: ResultWriter,
                 parameterRanges: Params,
                 combination_number: Optional[int] = None):
    hp_searcher = mt.HyperParameterSearcher(scoring='f1',
                                            data_source=data_source,
                                            result_writer=result_writer)

    parameters = parameterRanges.getParameters()
    for key, value in parameters.items():
        if not isinstance(value, List):
            parameters[key] = [value]

    hp_searcher.fit(estimator_name='Keras',
                    estimator=estimator,
                    parameters=parameters,
                    combination_number=combination_number)
