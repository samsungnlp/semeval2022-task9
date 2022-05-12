import unittest

from src.fetch_resources import fetch_linguistic_resources
from src.pipeline.answerers.location_change import QuestionAnswererLocationChange, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe, Q_A, QuestionAnswerRecipe


class TestQAClassLocationChange(unittest.TestCase):

    def setUp(self) -> None:
        fetch_linguistic_resources()

    def test_get_subject_from_question(self):
        with self.assertRaises(ValueError):
            QuestionAnswererLocationChange().get_subject_from_question("")

        with self.assertRaises(ValueError):
            QuestionAnswererLocationChange().get_subject_from_question("Where was X when Y")

        with self.assertRaises(ValueError):
            QuestionAnswererLocationChange().get_subject_from_question("Where will be X before Y")

        self.assertEqual("X_Y_Z", QuestionAnswererLocationChange().get_subject_from_question("Where was X Y Z  before Q"))
        self.assertEqual("X", QuestionAnswererLocationChange().get_subject_from_question("Where was the X before Q"))

        self.assertEqual("Y_Z", QuestionAnswererLocationChange().get_subject_from_question("Where was an Y Z before Q"))
        self.assertEqual("ZZ", QuestionAnswererLocationChange().get_subject_from_question("Where was a ZZ before Q"))

    def test_get_ref_verb(self):
        with self.assertRaises(ValueError):
            QuestionAnswererLocationChange.get_reference_verb_from_question("")

        with self.assertRaises(ValueError):
            QuestionAnswererLocationChange.get_reference_verb_from_question("Where was QQ?")

        with self.assertRaises(ValueError):
            QuestionAnswererLocationChange.get_reference_verb_from_question("Where was X before Y it will")

        self.assertEqual("Qed",
                         QuestionAnswererLocationChange.get_reference_verb_from_question("Where was X Y Z before it was Qed"))
        self.assertEqual("Q",
                         QuestionAnswererLocationChange.get_reference_verb_from_question("Where was X Y Z before it was Q R S T U"))

    def test_prediction(self):
        recipe = Recipe.return_recipe_for_test()
        q_a = Q_A.build_dummy_qa("Where was cooked vegetable before it was combined with meat?", "1-1", "pan")
        question = QuestionAnswerRecipe(q_a, recipe)

        engine = QuestionAnswererLocationChange()
        res = engine.answer_a_question(question, QuestionCategory("whatever"))
        self.assertIsInstance(res, PredictedAnswer)
        self.assertTrue(res.has_answer())
        self.assertEqual("pan", res.answer)
