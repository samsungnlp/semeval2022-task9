import unittest

from src.fetch_resources import fetch_linguistic_resources
from src.pipeline.answerers.counting_times import QuestionAnswererCountingTimes, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQAClass0HowManyTimes(unittest.TestCase):

    def setUp(self) -> None:
        fetch_linguistic_resources()

    def test_prediction(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("How many times is the pan used?", "question_id", "4")
        question = QuestionAnswerRecipe(q_a, recipe)

        res = QuestionAnswererCountingTimes().answer_a_question(question, QuestionCategory("question_id"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("4", res.answer)

    def test_prediction_not_found(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("How many times is the invalid something used?", "question_id", None)
        question = QuestionAnswerRecipe(q_a, recipe)

        res = QuestionAnswererCountingTimes().answer_a_question(question, QuestionCategory("question_id"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertFalse(res.has_answer())
