from src.pipeline.interface_question_answering import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe, Q_A
from src.pipeline.answerers.class_5 import QuestionAnswerer5

import unittest


class TestClass5Engine(unittest.TestCase):

    def test_invalid_question(self):
        engine = QuestionAnswerer5()

        question = QuestionAnswerRecipe(Q_A("# question 1-2 = What?"), recipe=None)
        res = engine.answer_a_question(question, QuestionCategory("5"))
        self.assertIsNone(res.answer)
        self.assertFalse(res.has_answer())
        self.assertIn("source", res.more_info)
