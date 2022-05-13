import unittest

from src.pipeline.interface_question_answering import PredictedAnswer


class TestPredictedAnswer(unittest.TestCase):

    def test_invalid(self):
        with self.assertRaises(ValueError):
            PredictedAnswer("")

        with self.assertRaises(ValueError):
            PredictedAnswer("Whatever", confidence=-1)

        with self.assertRaises(ValueError):
            PredictedAnswer("Whatever", confidence=2)

    def test_i_dont_know(self):
        res = PredictedAnswer(None, more_info={"source": "I don't know"})
        self.assertFalse(res.has_answer())

    def test_n_a(self):
        res = PredictedAnswer("N/A", raw_question="Is P == NP?")
        self.assertTrue(res.has_answer())
        self.assertEqual("Is P == NP?", res.raw_question)
