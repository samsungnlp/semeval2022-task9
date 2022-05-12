import unittest

from src.pipeline.answerers.method_preheat import QuestionAnswererMethodPreheat, PredictedAnswer, QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQuestionAnswererMethodPreheat(unittest.TestCase):
    def test_cannot_answer(self):
        engine = QuestionAnswererMethodPreheat()
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(Q_A.build_dummy_qa("How do you preheat the oven?", "whatever"), recipe)
        category = QuestionCategory("whatever")
        ret = engine.answer_a_question(question, category)
        self.assertIsInstance(ret, PredictedAnswer)
        self.assertFalse(ret.has_answer())
