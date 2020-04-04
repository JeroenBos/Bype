import unittest
from python.keyboard.hp import Params


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


if __name__ == '__main__':
    HpParamsTests().test_params_subclass_clone()
