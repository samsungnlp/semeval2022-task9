import unittest

from src.pipeline.answerers.class_14_preheat import QuestionAnswerer14Preheat, PredictedAnswer, QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQuestionAnswerer14Preheat(unittest.TestCase):
    def test_cannot_answer(self):
        engine = QuestionAnswerer14Preheat()
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(Q_A.build_dummy_qa("How do you preheat the oven?", 14), recipe)
        category = QuestionCategory("14_preheat")
        ret = engine.answer_a_question(question, category)
        self.assertIsInstance(ret, PredictedAnswer)
        self.assertFalse(ret.has_answer())
