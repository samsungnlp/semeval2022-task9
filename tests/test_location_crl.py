import unittest

from src.fetch_resources import fetch_linguistic_resources
from src.pipeline.answerers.location_crl import QuestionAnswererLocationCrl, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQAClassLocationCrl(unittest.TestCase):

    def setUp(self) -> None:
        fetch_linguistic_resources()

    def test_prediction(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("Where should you add the chopped vegetables?", "whatever", "pan")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswererLocationCrl()
        res = engine.answer_a_question(question, QuestionCategory("whatever"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("pan", res.answer)
