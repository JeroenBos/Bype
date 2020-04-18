import unittest

class Test(unittest.TestCase):
    def test_diagonal_indices(self):
        L = 3

        diagonals = [(i // L) * (L + 1) for i in range(L * L)]

        assert diagonals == [0, 0, 0, 4, 4, 4, 8, 8, 8]

    def test_alterinative_diagonal_indices(self):
        L = 3

        diagonals = [(i % L) * (L + 1) for i in range(L * L)]

        assert diagonals == [0, 4, 8, 0, 4, 8, 0, 4, 8]


if __name__ == '__main__':
    Test().test_diagonal_indices()
