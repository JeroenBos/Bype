import math
import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import generated_convolved_data as convolved_data, generated_data as data, generated_data_max_timestep as max_timesteps
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics, ValidationData
from keyboard._4_model import KeyboardEstimator
from keyboard._4b_initial_weights import ReloadWeights
from typing import List, Union
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler  # noqa
import pandas as pd
import MyBaseEstimator
from MyBaseEstimator import best_model_path
from os import path

MyBaseEstimator.global_phase = 1
MyBaseEstimator.global_run = 0


assert len(max_timesteps) == 1, 'not implemented'
preprocessor = Preprocessor(max_timesteps=list(max_timesteps)[0])


metric = Metrics(ValidationData(data, preprocessor))

weight_init_strategy = ReloadWeights(best_model_path)

training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=200,
                                                              weight_init_strategy=weight_init_strategy,
                                                              )   \
                                          .with_callback(metric)  \
                                          .fit(convolved_data)
