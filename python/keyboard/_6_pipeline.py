from dataclasses import dataclass
import math
import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import perfect_swipes, get_timesteps
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics, ValidationData
from keyboard._4_model import ModelFactory
from keyboard._4a_word_input_model import WordStrategy, CappedWordStrategy
from keyboard._4b_initial_weights import ReloadWeights
from typing import List, Union, Any, Optional, Iterable, Callable
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler  # noqa
import pandas as pd
from os import path
from trainer.trainer import TrainingsPlanBase
from trainer.types import TrainerExtension
from trainer.extensions.ContinuousEpochsCount import ContinuousEpochCountExtensions as EpochsKeepCounting, ApplyInitialEpochAndNumEpochToFitArgsTrainerExtension as ApplyInitialEpochAndNumEpochToFitArgs
from trainer.extensions.LoadInitialWeights import LoadInitialWeightsTrainerExtension as LoadInitialWeights
from trainer.extensions.MetricExtension import ValidationDataScoringExtensions as AddValidationDataScoresToTensorboard
from trainer.extensions.preprocessor import SetMaxTimestepTrainerExtension as SetMaxTimestep, ComputeSwipeFeatureCountTrainerExtension as ComputeSwipeFeatureCount, PreprocessorTrainerExtension as PreprocessorExtension
from trainer.extensions.TagWithTimestamp import TagWithTimestampTrainerExtension as TagWithTimestamp, LogDirPerDataTrainerExtension as LogDirPerData
from trainer.extensions.GenerateData import GenerateDataTrainerExtension as GenerateData
from trainer.extensions.tensorboard import TensorBoardExtension
from trainer.ModelAdapter import CompileArgs, FitArgs, ParameterizeModelExtension as ParameterizeModel
from trainer.extensions.fit_datasource import AllowDataSources
from trainer.extensions.SaveBestModel import SaveBestModelTrainerExtension as SaveBestModel

class Params:   
    tag: Optional[str] = None 
    # required by GenerateData:
    n_words: int
    n_chars: int
    verify: Optional[bool] = False
    # required by PreprocessorExtension:
    n_epochs: int
    word_input_strategy: WordStrategy
    #
    fit_args = FitArgs()
    compile_args = CompileArgs()
    log_dir: str = 'logs/'
    convolution_fraction: float = 1.0

    def __init__(self, **kw):
        mandatory = ['n_words', 'n_chars', 'word_input_strategy']
        for m in mandatory:
            assert m in kw, f"Missing mandatory argument '{m}'"

        self.__dict__.update(kw)

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if not name.startswith('__') and isinstance(value, Callable):
            return value()
        return value

    @property
    def best_model_path(self) -> str:
        """ The path at which the best model is saved. """
        return self.log_dir + 'best_model.h5'

# best_model_path ?


params = Params(
    n_epochs=100,
    n_words=10,
    n_chars=10,
    word_input_strategy=CappedWordStrategy(5),
    fit_args=FitArgs(
    ),
    compile_args=CompileArgs(
    ),
    max_timesteps=108,  # HACK
    filebased_continued_epoch_counting=True,

)


class TrainingsPlan(TrainingsPlanBase):
    @property
    def params(self) -> Iterable[Params]:
        yield params

    def get_extensions(self, params: Params, prev_params: Optional[Params]) -> Iterable[TrainerExtension]:
        # initialization and callback registration:
        yield TagWithTimestamp(params)
        yield LogDirPerData(params)
        yield TensorBoardExtension(params)
        yield EpochsKeepCounting(params, prev_params)
        yield ApplyInitialEpochAndNumEpochToFitArgs(params)
        yield AddValidationDataScoresToTensorboard(params)
        yield SaveBestModel(params)

        # data generation:
        yield GenerateData(params)
        yield SetMaxTimestep(params, prev_params)
        yield ComputeSwipeFeatureCount(params, prev_params)
        yield AllowDataSources()
        yield PreprocessorExtension(params)
        yield AllowDataSources()

        # model generation:
        yield ModelFactory(params)
        yield ParameterizeModel(params)
        yield LoadInitialWeights(params)

        


TrainingsPlan().execute()
