from src.pipeline.end_to_end_prediction import EndToEndQuestionAnsweringPrediction, PredictedAnswer, \
    QuestionAnswerRecipe

import unittest


class TestEndToEndQuestionAnsweringPrediction(unittest.TestCase):

    def test_load_val_dataset(self):
        engine = EndToEndQuestionAnsweringPrediction("val", False)
        res = engine.load_dataset(limit_recipes=10)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 300)
        self.assertTrue(all([isinstance(x, QuestionAnswerRecipe) for x in res]))

    def test_load_test_dataset(self):
        engine = EndToEndQuestionAnsweringPrediction("test", False)
        res = engine.load_dataset(limit_recipes=10)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 300)
        self.assertTrue(all([isinstance(x, QuestionAnswerRecipe) for x in res]))

    def test_load_train_dataset(self):
        engine = EndToEndQuestionAnsweringPrediction("train", False)
        res = engine.load_dataset(limit_recipes=2)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 27 + 57)
        self.assertTrue(all([isinstance(x, QuestionAnswerRecipe) for x in res]))

    def test_bad_dataset(self):
        with self.assertRaises(Exception):
            EndToEndQuestionAnsweringPrediction("bad dataset", False)

    def test_the_pipeline(self):
        engine = EndToEndQuestionAnsweringPrediction("val", False)
        engine.limit_recipes = 5
        questions, answers = engine.run_prediction()
        self.assertGreaterEqual(len(questions), 100)
        self.assertGreaterEqual(len(answers), 100)
        self.assertEqual(len(questions), len(answers))
        for answer in answers:
            self.assertIsInstance(answer, PredictedAnswer)
            self.assertIn(answer.answer, {None, "the first event", "1", "by using a knife"})
