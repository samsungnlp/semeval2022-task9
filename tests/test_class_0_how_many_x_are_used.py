import unittest

from src.fetch_resources import fetch_linguistic_resources
from src.pipeline.answerers.class_0 import QuestionAnswerer0, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQAClass0HowManyTimes(unittest.TestCase):

    def setUp(self) -> None:
        fetch_linguistic_resources()

    def test_prediction0_2(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("How many spatulas are used?", "0-2", "1")
        question = QuestionAnswerRecipe(q_a, recipe)

        res = QuestionAnswerer0().answer_a_question(question, QuestionCategory("0"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("1", res.answer)

    def test_prediction_not_found(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("How many invalid somethings are used?", "0-0", None)
        question = QuestionAnswerRecipe(q_a, recipe)

        res = QuestionAnswerer0().answer_a_question(question, QuestionCategory("0"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertFalse(res.has_answer())
