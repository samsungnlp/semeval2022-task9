from src.pipeline.interface_question_answering import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe, Q_A, Recipe
from src.pipeline.answerers.class_2_6_10_14 import QuestionAnswerer2_6_10_14

import unittest


class TestClass8Engine(unittest.TestCase):

    def test_invalid_question(self):
        engine = QuestionAnswerer2_6_10_14(
            ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"], ["TOOL"],
            ["Patient", "Theme"], ["Instrument"],
            ["Patient", "Theme"], ["Attribute"],
            ["Patient", "Theme"], ["Goal"]
        )
        q_a = Q_A.build_dummy_qa("Where was cooked vegetable before it was combined with meat?", "17 - 1", None)
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(q_a, recipe)
        res = engine.answer_a_question(question, QuestionCategory("8"))
        self.assertIsNone(res.answer)
        self.assertFalse(res.has_answer())

    def test_valid_question(self):
        engine = QuestionAnswerer2_6_10_14(
            ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"], ["TOOL"],
            ["Patient", "Theme"], ["Instrument"],
            ["Patient", "Theme"], ["Attribute"],
            ["Patient", "Theme"], ["Goal"]
        )
        q_a = Q_A.build_dummy_qa("How do you saute onion?", "6-0", None)
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(q_a, recipe)
        res = engine.answer_a_question(question, QuestionCategory("6"))
        self.assertEqual(res.answer, "saute onion in 2 tablespoons of olive oil")
        self.assertTrue(res.has_answer())
