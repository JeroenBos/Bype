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
from trainer.extensions.ContinuousEpochsCount import ContinuousEpochCountExtensions as EpochsKeepCounting, ApplyInitialEpochAndNumEpochToFitArgsTrainerExtension as ApplyInitialEpochAndNumEpochToFitArgs, Params as ContinuousEpochCountParams, ContinuousStageCountExtensions as StagesKeepCounting
from trainer.extensions.LoadInitialWeights import LoadInitialWeightsTrainerExtension as LoadInitialWeights
from trainer.extensions.MetricExtension import TotalValidationDataScoringExtensions, ValidationDataScoringExtensions as AddValidationDataScoresToTensorboard
from trainer.extensions.preprocessor import SetMaxTimestepTrainerExtension as SetMaxTimestep, ComputeSwipeFeatureCountTrainerExtension as ComputeSwipeFeatureCount, PreprocessorTrainerExtension as PreprocessorExtension, PreprocessorParams
from trainer.extensions.TagWithTimestamp import TagWithTimestampTrainerExtension as TagWithTimestamp, LogDirPerDataTrainerExtension as LogDirPerData, RunLogDirTrainerExtension as RunLogDir
from trainer.extensions.BalanceWeights import BalanceWeightsTrainerExtension as BalanceWeights
from trainer.extensions.GenerateData import GenerateDataTrainerExtension as GenerateData, Params as DataGenenerationParams
from trainer.extensions.tensorboard.tensorboard import TensorBoardExtension
from trainer.extensions.tensorboard.scalar import TensorBoardScalar
from trainer.extensions.tensorboard.ResourceWriterPool import Params as ResourceWriterPoolParams
from trainer.extensions.CountWeights import count_params, PrintWeightsCount
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
             ResourceWriterPoolParams,
             ParamsBase):
    tag: Optional[str] = None 
    log_dir: str = 'logs/nchar10nwords1000/'
    run_log_dir: str = 'logs/nchar10words1000uncapped/'
    continue_weights: bool = True

    @property
    def filebased_continued_stage_counting(self) -> bool:
        return True

    @property
    def filebased_continued_epoch_counting(self) -> bool:
        # defaults to whether the weights are continued in the training
        return self.continue_weights

    @property
    def best_model_path(self) -> str:
        """ Gets the path at which the best model is saved. """
        log_dir = self.log_dir if getattr(self, "best_model_of_all_runs", False) is True else self.run_log_dir
        return path.join(log_dir, 'best_model.h5')

    @property
    def initial_weights_path(self) -> str:
        """ Gets the path from which to load the initial weights to be applied to the model after compilation. """
        return self.best_model_path

    @property
    def max_timesteps(self):
        result = set(len(entry) for entry in self.data().swipes)
        return max(result)



class TrainingsPlan(TrainingsPlanBase):
    @property
    def params(self) -> Iterable[Params]:
        for i in range(20):
            yield Params(
                n_epochs=400,
                n_words=1000,
                n_chars=10,
                word_input_strategy=CappedWordStrategy(10),
                continue_weights=False,
            )

    def get_extensions(self, params: Params, prev_params: Optional[Params]) -> Iterable[Union[TrainerExtension, type(TrainerExtension)]]:
        # initialization and callback registration:
        yield TagWithTimestamp
        yield LogDirPerData
        yield StagesKeepCounting
        yield EpochsKeepCounting
        yield RunLogDir  # must be after StagesKeepCounting
        yield TensorBoardExtension
        yield ApplyInitialEpochAndNumEpochToFitArgs
        yield AddValidationDataScoresToTensorboard
        yield SaveBestModel(lambda params: params.best_model_path, monitor='test_loss')  # must be after AddValidation
        yield EarlyStopping(patience=5, monitor="loss", baseline=0.005)

        # data generation:
        yield GenerateData()
        yield AllowDataSources()
        yield PreprocessorExtension()
        yield BalanceWeights()

        # model generation:
        yield ModelFactory
        yield ParameterizeModel
        # if params.continue_weights:
        #    yield LoadInitialWeights(on_first_stage="/home/jeroen/git/bype/python/logs/2020_05_30/best_model.h5")

        yield TensorBoardScalar(stage=lambda params: params.stage, n_chars=lambda params: params.n_chars)
        yield PrintWeightsCount


TrainingsPlan().execute()
