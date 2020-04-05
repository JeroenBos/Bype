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

    def test_hpEstimator_default_override__get_param_names(self):
        estimator = AbstractHpEstimator(lambda params: TestMLModel(params), Params)
        result = estimator._get_param_names()
        assert len(result) == 2
        assert result[0] == 'activation'
        assert result[1] == 'num_epochs'

    # def test_hp_search(self):
    #     class HpEstimator(AbstractHpEstimator):
    #         @classmethod
    #         def _get_param_names(cls):
    #             return sorted(['only_param'])
    #     do_hp_search(HpEstimator(lambda params: TestMLModel(params), Params),
    #                  InMemoryDataSource(df, 0),
    #                  ResultOutputWriter(),
    #                  Params(

    def test_metaclass_indexer(self):
        ref_var = [0]

        class Meta(type):
            def __getitem__(self, key):
                ref_var[0] = 1
                return self

        class T(metaclass=Meta):
            pass

        Tprime = T[0]
        assert ref_var[0] == 1
        t = Tprime()
        assert isinstance(t, Tprime)
        assert isinstance(t, T)


if __name__ == '__main__':
    HpParamsTests().test_params_subclass_clone()
