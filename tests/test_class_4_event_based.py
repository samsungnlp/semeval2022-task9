import unittest

from src.fetch_resources import fetch_linguistic_resources
from src.pipeline.answerers.class_4_v2 import QuestionAnswerer4EventBased, PredictedAnswer, C4Event
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQAClass4EventBased(unittest.TestCase):

    def setUp(self) -> None:
        fetch_linguistic_resources()

    def test_prediction5(self):
        recipe = Recipe.return_recipe_for_test()
        q = "Cutting the stem into bite - size pieces and sauting minced meat in a separate pan, which comes first?"
        q_a = Q_A.build_dummy_qa(q, "4-5", "the first event")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer4EventBased()
        res = engine.answer_a_question(question, QuestionCategory("4"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("the first event", res.answer)

    def test_prediction2(self):
        recipe = Recipe.return_recipe_for_test()
        q = "Sauting minced meat in a separate pan and cutting carrots and zucchini into cubes, which comes first?"
        q_a = Q_A.build_dummy_qa(q, "4-2", "the second event")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer4EventBased()
        res = engine.answer_a_question(question, QuestionCategory("4"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("the second event", res.answer)

    def test_prediction1(self):
        recipe = Recipe.return_recipe_for_test()
        q = "Transferring the meat into the pan with the vegetables and adding chopped vegetables, which comes first?"
        q_a = Q_A.build_dummy_qa(q, "4-2", "the second event")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer4EventBased()
        res = engine.answer_a_question(question, QuestionCategory("4"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("the second event", res.answer)

    def test_prediction0(self):
        recipe = Recipe.return_recipe_for_test()
        q = "Transferring the meat into the pan with the vegetables and seasoning the meat, which comes first?"
        q_a = Q_A.build_dummy_qa(q, "4-2", "the first event")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer4EventBased()
        res = engine.answer_a_question(question, QuestionCategory("4"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("the first event", res.answer)

    def test_prediction4(self):
        recipe = Recipe.return_recipe_for_test()
        q = "Cutting carrots and zucchini into cubes and adding the tinned tomatoes, which comes first?"
        q_a = Q_A.build_dummy_qa(q, "4-4", "the first event")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswerer4EventBased()
        res = engine.answer_a_question(question, QuestionCategory("4"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("the first event", res.answer)

    def test_split_query_2events(self):
        q = "Cutting the stem into bite - size pieces and sauting minced meat in a " \
            "separate pan, which comes first?"

        engine = QuestionAnswerer4EventBased()
        res = engine.split_question_into_events(q)
        self.assertEqual(2, len(res))
        self.assertEqual("cutting", res[0][0])
        self.assertEqual("sauting", res[1][0])

    def test_split_query_3events(self):
        q = "Sauting minced meat in a separate pan and cutting carrots and zucchini into cubes, which comes first?"
        engine = QuestionAnswerer4EventBased()
        res = engine.split_question_into_events(q)
        self.assertEqual(2, len(res))
        self.assertEqual("sauting", res[0][0])
        self.assertEqual("cutting", res[1][0])

        q = "Adding soy sauce , chili sauce , tomato ketchup , salt , pepper powder and heating oil in a pan," \
            " which comes first?"
        res = engine.split_question_into_events(q)
        self.assertEqual(2, len(res))
        self.assertEqual("adding", res[0][0])
        self.assertNotIn("and", res[0])
        self.assertEqual("heating", res[1][0])

    def test_extract_event(self):
        engine = QuestionAnswerer4EventBased()
        res = engine.extract_events_from_segment(["cutting", "the", "stem", "into", "bite", "-", "size", "pieces"])
        self.assertIsInstance(res, C4Event)
        self.assertEqual("cut", res.verb)
        self.assertIn("stem", res.objects)
        self.assertIn("into", res.all_words)
        self.assertIn("size", res.all_words)

        res = engine.extract_events_from_segment(["sauting", "minced", "meat", "in", "a", "separate", "pan"])
        self.assertEqual("saute", res.verb)
        self.assertIn("minced_meat", res.objects)
        self.assertIn("in", res.all_words)
        self.assertIn("pan", res.all_words)
