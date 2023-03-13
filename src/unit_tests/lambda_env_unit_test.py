import unittest
import sys

sys.path.append("../")
from calculus_path_mod.term_engine import *
from calculus_path_mod.reduction_strategy import *
from calculus_path_mod.json_serialization import load_terms
from calculus_path_mod.path_utils import similar

# norm_strategy = RandomOuterStrategy(prob_norm="pow_2")
norm_strategy = LOStrategy()


class TestLambdaEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_terms_list = load_terms("terms_src/test_terms.dat")
        self.expected_terms_list = load_terms("terms_src/expected_terms.dat")

    def uni_term_test(self, term_index: int):
        term_index -= 1
        norm_term, norm_steps = self.test_terms_list[term_index].normalize(norm_strategy)
        self.assertTrue(similar(norm_term, self.expected_terms_list[term_index]))
        self.assertTrue(norm_term.funky_str() == self.expected_terms_list[term_index].funky_str())

    def test_term_1(self):
        self.uni_term_test(1)

    def test_term_2(self):
        self.uni_term_test(2)

    def test_term_3(self):
        self.uni_term_test(3)

    def test_term_4(self):
        self.uni_term_test(4)

    def test_term_5(self):
        self.uni_term_test(5)

    def test_term_6(self):
        self.uni_term_test(6)

    def test_term_7(self):
        self.uni_term_test(7)

    def test_term_8(self):
        self.uni_term_test(8)

    def test_term_9(self):
        self.uni_term_test(9)

    def test_term_10(self):
        self.uni_term_test(10)

    def test_term_11(self):
        self.uni_term_test(11)

    def test_term_12(self):
        self.uni_term_test(12)

    def test_term_13(self):
        self.uni_term_test(13)


if __name__ == "__main__":
    unittest.main()
