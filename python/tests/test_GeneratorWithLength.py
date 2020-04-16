import unittest
from GeneratorWithLength import GeneratorWithLength

class TestGeneratorWithLength(unittest.TestCase): 
    def test_add_with_list(self):
        a = GeneratorWithLength([0, 1, 2], 3)
        b = [3, 4, 5]

        ab = a + b
        assert len(ab) == 6
        assert ab != [0, 1, 2, 3, 4, 5]
        assert list(ab) == [0, 1, 2, 3, 4, 5]

    def test_add_with_other_generator_with_length(self):
        a = GeneratorWithLength([0, 1, 2], 3)
        b = GeneratorWithLength([3, 4, 5], 3)

        ab = a + b
        assert len(ab) == 6
        assert ab != [0, 1, 2, 3, 4, 5]
        assert list(ab) == [0, 1, 2, 3, 4, 5]


    def test_enumerating_multiple_times(self):
        a = GeneratorWithLength([0, 1, 2], 3)

        aa = a + a
        assert len(aa) == 6
        assert aa != [0, 1, 2, 0, 1, 2]
        assert list(aa) == [0, 1, 2, 0, 1, 2]

    def test_generator_from_function(self):
        a = GeneratorWithLength(lambda i: i, 3)

        assert len(a) == 3
        assert list(a) == [0, 1, 2]


    def test_generator_from_function_appended_to_self(self):
        a = GeneratorWithLength(lambda i: i, 3)

        aa = a + a
        assert len(aa) == 6
        assert aa != [0, 1, 2, 0, 1, 2]
        assert list(aa) == [0, 1, 2, 0, 1, 2]



    def test_generators(self):
        a = (i for i in range(0, 3))
        b = (i for i in range(3, 6))

        ab = (*a, *b)
        assert len(list(ab)) == 6

        count = 0
        for i in ab:
            count += 1
        assert count == 6


if __name__ == '__main__':
    TestGeneratorWithLength().test_generator_from_function_appended_to_self()


        
