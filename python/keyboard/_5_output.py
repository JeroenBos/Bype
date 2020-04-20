from tensorflow.keras.models import load_model, Model  # noqa
from MyBaseEstimator import get_log_dir
from os import path
import numpy as np
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._2_transform import Preprocessor
from keyboard._4_model import KeyboardEstimator
from keyboard.MyModelCheckpoint import get_processor_path
import json
from myjson import json_decoders
from utilities import memoize

most_recent_modelpath = get_log_dir('logs') + path.sep + 'model.h5'


def load_model_with_preprocessor(model_path: str = most_recent_modelpath) -> "LoadedKeyboardModel":
    if not os.path.exists(model_path):
        raise ValueError(f"Path '{model_path}' doesn't exist")
    

    with open(get_processor_path(model_path)) as file:    
        preprocessor_json = json.load(file)
    preprocessor = json_decoders[Preprocessor.__name__](preprocessor_json)

    model = load_model(model_path)

    return LoadedKeyboardModel(model, preprocessor)

class LoadedKeyboardModel:
    def __init__(self, model: Model, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.model = model


    def _preprocess(self, X: SwipeEmbeddingDataFrame) -> np.ndarray:
        if self.preprocessor is None:
            return X
        return self.preprocessor._preprocess(X)

    def predict(self, X: SwipeEmbeddingDataFrame) -> np.ndarray:
        preprocessedX = self._preprocess(X)
        return self.model.predict(preprocessedX)


@memoize
def most_recent_model():
    return load_model_with_preprocessor(most_recent_modelpath)


class Scoring:
