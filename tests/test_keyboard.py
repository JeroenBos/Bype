import unittest
from python.keyboard.hp import Params


class HpParamsTests(unittest.TestCase):
    def test_params_clone(self):
        class TestParams(Params):
            pass
        params = TestParams()
        paramsAsDict = params.getParameters()
        assert len(paramsAsDict) == 2
        assert 'activation' in paramsAsDict
        assert 'num_epochs' in paramsAsDict


if __name__ == '__main__':
    HpParamsTests().test_params_clone()
