import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback  # noqa
from sklearn.metrics import roc_auc_score


class Metrics(Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        # X_val, y_val = self.validation_data[0], self.validation_data[1]
        # y_predict = np.asarray(self.model.predict(X_val))
  
        # y_val = np.argmax(y_val, axis=1)
        # y_predict = np.argmax(y_predict, axis=1)
        print(' - metric: 0 %')
        self._data.append({
            'val_rocauc': 1,
        })
        return

    def get_data(self):
        return self._data
