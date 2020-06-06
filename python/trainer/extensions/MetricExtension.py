import tensorflow as tf
from tensorflow.keras.callbacks import Callback  # noqa
from tensorflow.keras.models import Model  # noqa
from keyboard._3_scoring import Metrics, ValidationData
from trainer.trainer import TrainerExtension
from trainer.extensions.GenerateData import _unmemoized_generateDataTrainerExtension_compute_validation_data
from utilities import override


class ValidationDataScoringExtensions(TrainerExtension): 
    def __init__(self,
                 monitor_namespace: str = "",
                 print_loss=False,
                 print_misinterpretation_examples=False,
                 ):
        self._kw = {
            "monitor_namespace": monitor_namespace,
            "print_loss": print_loss,
            "print_misinterpretation_examples": print_misinterpretation_examples,
        }

    def initialize(self):
        # reserve a spot in the fit_args.callbacks (order is relevant)
        # only later fill it with the actual callback, which needs more information than is available at this point (the data)
        self.placeholder = CallbackPlaceholder()
        self.params.fit_args.callbacks.append(self.placeholder)

    def _get_data(self) -> ValidationData:
        return self.params.validation_data

    def after_compile(self, model: Model) -> None:
        callback = Metrics(
            validation_data=self._get_data(), 
            write_scalar=self.params.write_scalar,
            **self._kw
        )
        assert hasattr(self, "placeholder"), "You didn't call super().initialize()"
        self.placeholder.replace(self.params, callback)

    def _tf_summary_writer(self, i) -> tf.summary.SummaryWriter:
        return self.params.get_resource_writer(self.params.log_dir, i)




class TotalValidationDataScoringExtensions(ValidationDataScoringExtensions): 

    @property
    def _name(self):
        return "_total_validation_data" + self._kw["monitor_namespace"]


    def initialize(self):
        super().initialize()
        if self.is_first_stage:
            # we duplicate the cached version, because it's going to be modified
            validation_data = _unmemoized_generateDataTrainerExtension_compute_validation_data(self.params.data(), self.params.preprocessor)
            setattr(self.params, self._name, validation_data)
        else:
            validation_data = getattr(self.prev_params, self._name)
            validation_data.add(self.params.data(), self.params.preprocessor)
            setattr(self.params, self._name, validation_data)

    @override
    def _get_data(self):
        return getattr(self.params, self._name)


class CallbackPlaceholder(Callback):
    def __init__(self):
        super().__init__()
        self._replaced = False

    def replace(self, params, callback: Callback):
        assert not self._replaced, "Placeholder already replaced"
        self._replaced = True
        i = params.fit_args.callbacks.index(self)
        params.fit_args.callbacks[i] = callback

    def on_train_begin(self, batch, logs={}):
        raise ValueError("The placeholder should have been replaced")
