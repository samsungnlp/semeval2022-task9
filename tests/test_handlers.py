from src.pipeline.handlers import HandlerSaveToJson, PredictedAnswer
from src.unpack_data import QuestionAnswerRecipe, Recipe, Q_A
import json
import unittest
import tempfile
import pathlib


class TestJsonHandler(unittest.TestCase):

    def test_mismatching_lists(self):
        engine = HandlerSaveToJson("./bad/file")
        with self.assertRaises(ValueError):
            engine.handle_questions_answers([], ["ups", "mismatching", "lists", "lengths"])

    def test_dump_to_json(self):
        recipe = Recipe(["# newdoc id = 1234", "# newpar id = 1234::ingredients"])
        dummy_questions = [
            QuestionAnswerRecipe(qa=Q_A("# question 0-1 = q1?"), recipe=recipe),
            QuestionAnswerRecipe(qa=Q_A("# question 4-2 = q2?"), recipe=recipe),
            QuestionAnswerRecipe(qa=Q_A("# question 18-3 = q3?"), recipe=recipe),
        ]
        dummy_answers = [
            PredictedAnswer("8"),
            PredictedAnswer("the second event"),
            PredictedAnswer("N/A")
        ]

        with tempfile.TemporaryDirectory() as dir:
            filename = f"{dir}/output.json"
            engine = HandlerSaveToJson(filename)
            engine.handle_questions_answers(dummy_questions, dummy_answers)

            self.assertTrue(pathlib.Path(filename).exists())
            with open(filename) as f:
                as_json = json.load(f)

            expected = {"1234": {"0-1": "8", "4-2": "the second event", "18-3": "N/A"}}
            self.assertEqual(expected, as_json)
