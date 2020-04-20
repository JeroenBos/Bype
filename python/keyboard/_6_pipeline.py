import math
import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import generated_convolved_data as convolved_data, generated_data as data
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics
from keyboard._4_model import KeyboardEstimator
from typing import List, Union
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler  # noqa
import pandas as pd

preprocessor = Preprocessor(max_timesteps=3)

metric = Metrics(preprocessor.preprocess(convolved_data), preprocessor.decode, convolved_data.get_i, len(data))

loss_history = metric.losses
def adapt_learning_rate(epoch: int) -> float:
    def is_significantly_different(f: float) -> bool:
        return math.abs(f - 0.5) > 0.05 
    if(len(loss_history) < 2):
        return 0.05

    numbers_series = pd.Series(loss_history).rolling(3).mean()
    if is_significantly_different(numbers_series[-1]):
        epochs_since_last_significat_difference = len(numbers_series)
        for i in range(len(numbers_series)):
            if not is_significantly_different(numbers_series[-i - 1]):
                epochs_since_last_significat_difference = i
                break

        if epochs_since_last_significat_difference < 10:
            return 0.05
        elif epochs_since_last_significat_difference < 50:
            return 0.1
        elif epochs_since_last_significat_difference < 100:
            return 0.25
        else:
            return 0.5

    epochs_since_last_insignificat_difference = len(numbers_series)
    for i in range(len(numbers_series)):
        if is_significantly_different(numbers_series[-i - 1]):
            epochs_since_last_insignificat_difference = i
            break

    if epochs_since_last_insignificat_difference < 10:
        return 0.01
    elif epochs_since_last_insignificat_difference < 20:
        return 0.001
    elif epochs_since_last_insignificat_difference < 80:
        return 0.0001
    else:
        return 0.00001


lr_scheduler = LearningRateScheduler(adapt_learning_rate)


training = KeyboardEstimator[preprocessor].create_initialized(num_epochs=100)    \
                                          .with_callback(metric, lr_scheduler)   \
                                          .fit(convolved_data)
