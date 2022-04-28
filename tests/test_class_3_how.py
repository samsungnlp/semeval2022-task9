import unittest

from src.pipeline.interface_question_answering import QuestionCategory, PredictedAnswer
from src.pipeline.answerers.class_3_how import QuestionAnswerer3How
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQuestionAnswerer3How(unittest.TestCase):

    def test_prediction(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("How did you get the cooked vegetable?", "3-3", None)
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer3How()
        res = engine.answer_a_question(question, QuestionCategory("3_how"))
        self.assertEqual(res.answer, "by adding the chopped vegetables to the pan")
        self.assertTrue(res.has_answer())

    def test_no_answer_question(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("How did you get the mixture?", "18-0", None)
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer3How()
        res = engine.answer_a_question(question, QuestionCategory("3_how"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertFalse(res.has_answer())
