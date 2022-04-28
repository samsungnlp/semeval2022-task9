from src.pipeline.interface_question_answering import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe, Q_A, convert_train_data
from src.pipeline.answerers.class_3_what import QuestionAnswerer3What

import unittest


class TestClass3WhatEngine(unittest.TestCase):
    def setUp(self):
        self.engine = QuestionAnswerer3What()
        self.recipe = convert_train_data(limit_recipes=1)[0]

    def test_no_answer(self):
        question = QuestionAnswerRecipe(Q_A("# question 1-2 = What's in the soup?"),
                                        recipe=self.recipe)
        res = self.engine.answer_a_question(question, QuestionCategory("3_what"))

        self.assertIsNone(res.answer)
        self.assertFalse(res.has_answer())
        self.assertIn("source", res.more_info)
