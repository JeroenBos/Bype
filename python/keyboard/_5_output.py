import os
from tensorflow.keras.models import load_model, Model  # noqa
from os import path
import numpy as np
from keyboard._0_types import SwipeEmbeddingDataFrame, ProcessedInputSeries
from keyboard._1a_generate import generated_convolved_data as convolved_data, generated_data as data
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics
from keyboard._4_model import KeyboardEstimator
from trainer.extensions.SavePreprocessorJson import get_processor_path
import json
from myjson import json_decoders
from utilities import memoize


def load_model_with_preprocessor(model_path: str) -> "LoadedKeyboardModel":
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


    def predict(self, preprocessedX: ProcessedInputSeries) -> np.ndarray:
        return self.model.predict(preprocessedX)


    def _preprocess(self, X: SwipeEmbeddingDataFrame) -> ProcessedInputSeries:
        if self.preprocessor is None:
            return X
        return self.preprocessor._preprocess(X)

    def preprocess_and_predict(self, X: SwipeEmbeddingDataFrame) -> np.ndarray:
        preprocessedX = self._preprocess(X)
        return self.predict(preprocessedX)


# @memoize
# def most_recent_model():
#     return load_model_with_preprocessor(most_recent_modelpath)


# model = load_model_with_preprocessor(os.getcwd() + '/logs/2020_04_20/model.h5')

# preprocessor = Preprocessor(max_timesteps=2)

# metric = Metrics(preprocessor.preprocess(convolved_data), preprocessor.decode, convolved_data.get_i, len(data), model=model)

# metric.print_misinterpreted_words()
