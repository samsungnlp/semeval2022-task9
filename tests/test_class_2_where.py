import unittest

from src.fetch_resources import fetch_linguistic_resources
from src.pipeline.answerers.class_2_where import QuestionAnswerer2Where, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQAClass2Where(unittest.TestCase):

    def setUp(self) -> None:
        fetch_linguistic_resources()

    def test_prediction(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("Where should you add the chopped vegetables?", "2-2", "pan")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer2Where()
        res = engine.answer_a_question(question, QuestionCategory("2_where"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("pan", res.answer)
