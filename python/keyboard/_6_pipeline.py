import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import generated_convolved_data as convolved_data, generated_data as data
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics
from keyboard._4_model import KeyboardEstimator
from typing import List, Union
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler  # noqa

preprocessor = Preprocessor(max_timesteps=3)

metric = Metrics(preprocessor.preprocess(convolved_data), preprocessor.decode, convolved_data.get_i, len(data))
def adapt_learning_rate(epoch: int) -> float:
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.001
    elif epoch < 80:
        return 0.0001
    else:
        return 0.00001


lr_scheduler = LearningRateScheduler(adapt_learning_rate)


training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=100)    \
                                          .with_callback(metric, lr_scheduler)   \
                                          .fit(convolved_data)
