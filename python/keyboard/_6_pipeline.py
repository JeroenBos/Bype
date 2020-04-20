import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import generated_convolved_data as convolved_data, generated_data as data
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics
from keyboard._4_model import KeyboardEstimator
from typing import List, Union
from time import time

preprocessor = Preprocessor(max_timesteps=2)

metric = Metrics(preprocessor.preprocess(convolved_data), preprocessor.decode, convolved_data.get_i, len(data))

training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=1000)  \
                                          .with_callback(metric)               \
                                          .fit(convolved_data)
