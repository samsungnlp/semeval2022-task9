from src.pipeline.interface_question_answering import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe, Q_A, Recipe
from src.pipeline.answerers.universal_srl import QuestionAnswererUniversalSrl

import unittest


class TestEngineUniversalSrl(unittest.TestCase):

    def test_invalid_question(self):
        engine = QuestionAnswererUniversalSrl(["Patient", "Theme"],
                                              ["Location", "Destination", "Co-Patient", "Co-Theme"])
        q_a = Q_A.build_dummy_qa("Where was cooked vegetable before it was combined with meat?", "any_category", None)
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(q_a, recipe)
        res = engine.answer_a_question(question, QuestionCategory("any_category"))
        self.assertIsNone(res.answer)
        self.assertFalse(res.has_answer())

    def test_valid_question(self):
        engine = QuestionAnswererUniversalSrl(["Patient", "Theme"],
                                              ["Location", "Destination", "Co-Patient", "Co-Theme"])
        q_a = Q_A.build_dummy_qa("Where do you saute minced meat?", "any_category", None)
        recipe = Recipe.return_recipe_for_test()
        question = QuestionAnswerRecipe(q_a, recipe)
        res = engine.answer_a_question(question, QuestionCategory("any_category"))
        self.assertEqual(res.answer, "in a separate pan")
        self.assertTrue(res.has_answer())
