import pathlib
import tempfile
import unittest
from io import StringIO
from src.pipeline.handler_metrics_per_category import HandlerMetricsPerCategory, PredictedAnswer, Result
from src.unpack_data import Q_A, QuestionAnswerRecipe


class TestHandlerMetricsPerCategory(unittest.TestCase):

    def test_handle_bad_category(self):
        engine = HandlerMetricsPerCategory()
        engine.results_by_category["a_category"] = []

        with self.assertRaises(ValueError):
            engine.handle_category("bad_category")

    def test_handle_get_result_one_line(self):
        question = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "2", "answer"), recipe=None)
        answer = PredictedAnswer("answer")
        engine = HandlerMetricsPerCategory()
        engine.results_by_category["a_category"] = [Result(question, answer)]
        res = engine._get_results("a_category")

        self.assertIsInstance(res, list)
        first = res[0]
        self.assertIsInstance(first, dict)

        self.assertAlmostEqual(1.0, first["F1"])
        self.assertAlmostEqual(1.0, first["Exact Match"])
        self.assertEqual("answer", first["Actual Answer"])
        self.assertEqual("answer", first["Predicted Answer"])
        self.assertEqual("Q?", first["Question"])
        self.assertIn("Passage", first)

    def test_handle_question_witout_answer(self):
        question = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "2"), recipe=None)
        answer = PredictedAnswer("answer")
        engine = HandlerMetricsPerCategory()
        engine.results_by_category["13"] = [Result(question, answer)]
        res = engine._get_results("13")

        self.assertIsInstance(res, list)
        self.assertEqual(1, len(res))
        first = res[0]
        self.assertIsInstance(first, dict)

        self.assertIsNone(first["F1"])
        self.assertIsNone(first["Exact Match"])
        self.assertIsNone(first["Actual Answer"])
        self.assertEqual("answer", first["Predicted Answer"])
        self.assertEqual("Q?", first["Question"])
        self.assertIn("Passage", first)

    def test_group(self):
        """
        helper test for internal implementation
        """
        question = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "2", "answer"), recipe=None)
        answer1 = PredictedAnswer("answer1", more_info={"predicted_category": "category 1"})
        answer2 = PredictedAnswer("answer2", more_info={"predicted_category": "category 2"})
        answer3 = PredictedAnswer("answer3", more_info={"predicted_category": "category 1"})

        engine = HandlerMetricsPerCategory()
        engine._group_by_category([question, question, question], [answer1, answer2, answer3], more_info={})

        grouped = engine.results_by_category

        self.assertEqual(2, len(grouped))

        self.assertIn("category 1", grouped)
        self.assertIn("category 2", grouped)
        self.assertEqual(2, len(grouped["category 1"]))
        self.assertEqual(1, len(grouped["category 2"]))

    def test_handle_whole_category(self):
        question1 = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "1", "good answer"), recipe=None)
        question2 = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "2", "incorrect"), recipe=None)
        question3 = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "3", "good answer"), recipe=None)

        answer1 = PredictedAnswer("good answer", more_info={"predicted_category": "11_12"})
        answer2 = PredictedAnswer("bad answer", more_info={"predicted_category": "11_12"})
        answer3 = PredictedAnswer("good answer", more_info={"predicted_category": "11_12"})

        with tempfile.TemporaryDirectory() as dir:
            sink = StringIO()
            engine = HandlerMetricsPerCategory(prefix_dir=dir, outstream=sink)
            output_path = f"{dir}/results_category_11_12.xlsx"
            self.assertFalse(pathlib.Path(output_path).exists())

            engine.handle_questions_answers([question1, question2, question3], [answer1, answer2, answer3],
                                            more_info={})

            self.assertTrue(pathlib.Path(output_path).exists())
            log = sink.getvalue()
            self.assertRegex(log, "[Cc]at.*11_12")
            self.assertRegex(log, "[Cc]ount 3")
            self.assertRegex(log, "EM .* 0\\.66")
            self.assertRegex(log, "F1 .* 0\\.66")

    def test_handle_whole_category_no_answers(self):
        question1 = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "1"), recipe=None)
        question2 = QuestionAnswerRecipe(Q_A.build_dummy_qa("Q?", "2"), recipe=None)

        answer1 = PredictedAnswer("good answer", more_info={"predicted_category": "13"})
        answer2 = PredictedAnswer("bad answer", more_info={"predicted_category": "13"})

        with tempfile.TemporaryDirectory() as dir:
            sink = StringIO()
            engine = HandlerMetricsPerCategory(prefix_dir=dir, outstream=sink)

            engine.handle_questions_answers([question1, question2], [answer1, answer2],
                                            more_info={})

            output_path = f"{dir}/results_category_13.xlsx"
            self.assertTrue(pathlib.Path(output_path).exists())
            log = sink.getvalue()
            self.assertRegex(log, "EM = None")
            self.assertRegex(log, "F1 = None")
