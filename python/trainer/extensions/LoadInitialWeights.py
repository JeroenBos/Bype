from abc import ABC, abstractmethod
import logging
import os
from tensorflow.keras.models import load_model, Model  # noqa
from typing import Callable, Optional, Tuple, Union

from KerasModelPadder import copy_weights
from trainer.types import TrainerExtension, IModel
from utilities import override


class LoadInitialWeightsTrainerExtension(TrainerExtension):
    def __init__(self, on_first_stage: Union[bool, str] = True):
        """ 
        :param on_first_stage: Depending on the value, determines what is loaded in the first stage:
        - a string: is read as path and loaded
        - True: params.initial_weights_path (like all other stages)
        - False: nothing
        """
        assert isinstance(on_first_stage, (str, bool))
        self._on_first_stage = on_first_stage

    @property
    def _initial_weights_path(self):
        return getattr(self.params, "initial_weights_path", None)

    @override
    def create_model(self, model: Optional[IModel]) -> Optional[IModel]:
        assert model is not None, "A model must exist to load weights into it"
        assert self._initial_weights_path is not None, "No initial weights path provided"

        if not self.is_first_stage or self._on_first_stage is not False:
            path = self._on_first_stage if isinstance(self._on_first_stage, str) else self._initial_weights_path
            try:
                if not os.path.exists(path):
                    logging.error(f"File '{path}' does not exist")
            except Exception as e:
                logging.error(f"Error in loading file '{path}'")
                logging.error(e)

            # Load the h5 file into the specified model:
            copy_weights(model, path)
        return model
