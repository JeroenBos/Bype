import math
import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import perfect_swipes, get_timesteps
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

MyBaseEstimator.global_phase = 0
MyBaseEstimator.global_run = 0
verify = True
n_words = 100
n_chars = 10


data = SwipeEmbeddingDataFrame.__as__(perfect_swipes(n_words=n_words, n_chars=n_chars), verify=verify) 
convolved_data = data.convolve(fraction=1, verify=verify)



assert len(get_timesteps(convolved_data)) == 1, 'not implemented'
preprocessor = Preprocessor(max_timesteps=list(get_timesteps(convolved_data))[0])


metric = Metrics(ValidationData(data, preprocessor))

weight_init_strategy = ReloadWeights(best_model_path)

training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=1000,
                                                              weight_init_strategy=weight_init_strategy,
                                                              )   \
                                          .with_callback(metric)  \
                                          .fit(convolved_data)
