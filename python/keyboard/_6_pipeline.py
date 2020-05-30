from unordereddataclass import mydataclass
import math
import init_seed
from keyboard._0_types import SwipeEmbeddingDataFrame
from keyboard._1a_generate import perfect_swipes, get_timesteps
from keyboard._2_transform import Preprocessor
from keyboard._3_scoring import Metrics, ValidationData
from keyboard._4_model import ModelFactory, Params as CreateModelParams
from keyboard._4a_word_input_model import WordStrategy, CappedWordStrategy
from typing import List, Union, Any, Optional, Iterable, Callable
from time import time
from tensorflow.keras.callbacks import LearningRateScheduler  # noqa
import pandas as pd
from os import path
from trainer.trainer import TrainingsPlanBase
from trainer.types import TrainerExtension
from trainer.ModelAdapter import ParamsBase
from trainer.extensions.ContinuousEpochsCount import ContinuousEpochCountExtensions as EpochsKeepCounting, ApplyInitialEpochAndNumEpochToFitArgsTrainerExtension as ApplyInitialEpochAndNumEpochToFitArgs, Params as ContinuousEpochCountParams
from trainer.extensions.LoadInitialWeights import LoadInitialWeightsTrainerExtension as LoadInitialWeights
from trainer.extensions.MetricExtension import ValidationDataScoringExtensions as AddValidationDataScoresToTensorboard
from trainer.extensions.preprocessor import SetMaxTimestepTrainerExtension as SetMaxTimestep, ComputeSwipeFeatureCountTrainerExtension as ComputeSwipeFeatureCount, PreprocessorTrainerExtension as PreprocessorExtension, PreprocessorParams
from trainer.extensions.TagWithTimestamp import TagWithTimestampTrainerExtension as TagWithTimestamp, LogDirPerDataTrainerExtension as LogDirPerData
from trainer.extensions.BalanceWeights import BalanceWeightsTrainerExtension as BalanceWeights
from trainer.extensions.GenerateData import GenerateDataTrainerExtension as GenerateData, Params as DataGenenerationParams
from trainer.extensions.tensorboard import TensorBoardExtension
from trainer.ModelAdapter import CompileArgs, FitArgs, ParameterizeModelExtension as ParameterizeModel
from trainer.extensions.fit_datasource import AllowDataSources
from trainer.extensions.SaveBestModel import SaveBestModelTrainerExtension as SaveBestModel
from trainer.extensions.EarlyStopping import EarlyStoppingTrainerExtension as EarlyStopping, Params as EarlyStoppingParams



@mydataclass
# TODO: design mydataclass such that all base classes are considered to be dataclasses too
class Params(DataGenenerationParams, 
             PreprocessorParams, 
             CreateModelParams,
             ContinuousEpochCountParams,
             EarlyStoppingParams,
             ParamsBase):
    tag: Optional[str] = None 
    log_dir: str = 'logs/'

    @property
    def best_model_path(self) -> str:
        """ The path at which the best model is saved. """
        return self.log_dir + 'best_model.h5'

    @property
    def initial_weights_path(self) -> str:
        """ The path from which to load the initial weights to be applied to the model after compilation. """
        return self.best_model_path

    @property
    def max_timesteps(self):
        result = set(len(entry) for entry in self.data().swipes)
        assert len(result) == 1, "Multiple timestep lengths not implemented"
        return next(iter(result))

    


params = Params(
    n_epochs=100,
    n_words=10,
    n_chars=10,
    word_input_strategy=CappedWordStrategy(5),
    filebased_continued_epoch_counting=True,
)


class TrainingsPlan(TrainingsPlanBase):
    @property
    def params(self) -> Iterable[Params]:
        yield params

    def get_extensions(self, params: Params, prev_params: Optional[Params]) -> Iterable[Union[TrainerExtension, type(TrainerExtension)]]:
        # initialization and callback registration:
        yield TagWithTimestamp
        yield LogDirPerData
        yield TensorBoardExtension
        yield EpochsKeepCounting
        yield ApplyInitialEpochAndNumEpochToFitArgs
        yield AddValidationDataScoresToTensorboard
        yield SaveBestModel
        yield EarlyStopping(monitor="loss")

        # data generation:
        yield GenerateData()
        yield ComputeSwipeFeatureCount()
        yield AllowDataSources()
        yield PreprocessorExtension()
        yield BalanceWeights()

        # model generation:
        yield ModelFactory
        yield ParameterizeModel
        yield LoadInitialWeights




TrainingsPlan().execute()
