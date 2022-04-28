from src.pipeline.interface_question_answering import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe, Q_A, Recipe
from src.pipeline.answerers.class_8 import QuestionAnswerer8

import unittest


class TestClass8Engine(unittest.TestCase):

    def test_invalid_question(self):
        engine = QuestionAnswerer8(["Patient", "Theme"], ["Location", "Destination", "Co-Patient", "Co-Theme"])
        q_a = Q_A.build_dummy_qa("Where was cooked vegetable before it was combined with meat?", "17 - 1", None)
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(q_a, recipe)
        res = engine.answer_a_question(question, QuestionCategory("8"))
        self.assertIsNone(res.answer)
        self.assertFalse(res.has_answer())

    def test_valid_question(self):
        engine = QuestionAnswerer8(["Patient", "Theme"], ["Location", "Destination", "Co-Patient", "Co-Theme"])
        q_a = Q_A.build_dummy_qa("Where do you saute minced meat?", "8-1", None)
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(q_a, recipe)
        res = engine.answer_a_question(question, QuestionCategory("8"))
        self.assertEqual(res.answer, "in a separate pan")
        self.assertTrue(res.has_answer())
