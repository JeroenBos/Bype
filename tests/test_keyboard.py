import unittest
from python.keyboard.hp import Params, MLModel, AbstractHpEstimator


class TestMLModel(MLModel):
    def _create_model(self):
        return None


class HpParamsTests(unittest.TestCase):
    def test_params_clone(self):
        params = Params()
        paramsAsDict = params.getParameters()
        assert len(paramsAsDict) == 2
        assert 'activation' in paramsAsDict
        assert 'num_epochs' in paramsAsDict

    def test_params_subclass_clone(self):
        class TestParams(Params):
            def __init__(self,
                         num_epochs=5,
                         activation='relu',
                         test_param=''):
                super().__init__(num_epochs, activation)
                self.test_param = test_param

        params = TestParams()
        paramsAsDict = params.getParameters()
        assert len(paramsAsDict) == 3
        assert 'activation' in paramsAsDict
        assert 'num_epochs' in paramsAsDict
        assert 'test_param' in paramsAsDict

    def test_hpEstimator(self):
        estimator = AbstractHpEstimator(lambda params: TestMLModel(params), Params)
        estimator.set_params({'activation': 'sigmoid'})
        estimator.set_params({'activation': 'relu'})

        assert len(estimator.models) == 2

    def test_hpEstimator_overriding_classmethod(self):
        class HpEstimator(AbstractHpEstimator):
            @classmethod
            def _get_param_names(cls):
                return sorted([])

        estimator = HpEstimator(lambda params: TestMLModel(params), Params)
        result = estimator._get_param_names()
        assert len(result) == 0


if __name__ == '__main__':
    HpParamsTests().test_params_subclass_clone()
