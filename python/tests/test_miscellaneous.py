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

    def test_square_indices(self):
        for L in [4, 5, 6]:
            N = L * L
            for i in range(N):
                assert is_correct(L, N, i) == ((i % L) == 0)

    def test_2_1(self):
        L = 2
        N = 3
        assert is_correct(L, N, 0)
        assert not is_correct(L, N, 1)
        assert is_correct(L, N, 2)


    
    def test_3_1(self):
        L = 3
        N = 4
        assert is_correct(L, N, 0)
        assert not is_correct(L, N, 1)
        assert is_correct(L, N, 2)
        assert is_correct(L, N, 3)

    def test_line(self):
        for L in [1, 4, 5, 6]:
            N = L
            for i in range(N):
                assert is_correct(L, N, i)

        

def is_correct(L, N, i):
    return i * L // N != ((i - 1) * L) // N


if __name__ == '__main__':
    Test().test_line()
