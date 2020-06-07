import unittest
import numpy as np
from utilities import append_masked

NaN = 999

class TestMaskArrayAppend(unittest.TestCase):
    def test_a(self):
        a = np.ma.array([[1], [2], [3], [4]],
                        mask=[[True], [True], [True], [False]])

        result = append_masked(a, a)
        assert result.shape == (8, 1)

        a = np.ma.array([[1], [2], [3], [4]],
                        mask=[[True], [True], [True], [False]])

        result = append_masked(a, a, axis=1)
        assert result.shape == (4, 2)
