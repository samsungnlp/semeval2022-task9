import unittest

from src.data_statistics import QuestionsAnswersMinMaxAvgStats, PassageStatsCalculator, DataItem


class TestDataStatistics(unittest.TestCase):

    def test_empty(self):
        engine = QuestionsAnswersMinMaxAvgStats()
        ret = engine.calc_statistics([])
        self.assertEqual(0, ret["num_questions"])
        self.assertEqual(0, ret["num_answers"])
        self.assertIsNone(ret["avg_question_len"])
        self.assertIsNone(ret["avg_answer_len"])

    def test_one_item(self):
        engine = QuestionsAnswersMinMaxAvgStats()
        ret = engine.calc_statistics([DataItem("empty", "This is a question?", answer="This is an answer", id=None)])
        self.assertEqual(1, ret["num_questions"])
        self.assertEqual(1, ret["num_answers"])
        self.assertEqual(0, ret["num_empty_answers"])
        self.assertEqual(0, ret["num_na_answers"])
        self.assertEqual(0, ret["num_int_answers"])
        self.assertEqual(1, ret["num_textual_answers"])

        self.assertEqual(19, ret["min_question_len"])
        self.assertEqual(19, ret["max_question_len"])
        self.assertAlmostEqual(19, ret["avg_question_len"])

        self.assertEqual(17, ret["min_answer_len"])
        self.assertEqual(17, ret["max_answer_len"])
        self.assertAlmostEqual(17, ret["avg_answer_len"])

    def test_event_answer(self):
        engine = QuestionsAnswersMinMaxAvgStats()
        ret = engine.calc_statistics([DataItem("empty", "q?", answer="the first event", id=None),
                                      DataItem("empty", "q?", answer="the second event", id=None)])
        self.assertEqual(2, ret["num_first_second_event_answers"])
        self.assertEqual(0, ret["num_textual_answers"])

    def test_two_items(self):
        engine = QuestionsAnswersMinMaxAvgStats()
        ret = engine.calc_statistics([
            DataItem("empty", "q1", answer="3", id="1"),
            DataItem("empty", "q2", answer="N/A", id="1"),
            DataItem("empty", "q33", id="1")

        ])
        self.assertEqual(3, ret["num_questions"])
        self.assertEqual(2, ret["num_answers"])
        self.assertEqual(1, ret["num_empty_answers"])
        self.assertEqual(1, ret["num_na_answers"])
        self.assertEqual(1, ret["num_int_answers"])
        self.assertEqual(0, ret["num_textual_answers"])

        self.assertEqual(2, ret["min_question_len"])
        self.assertEqual(3, ret["max_question_len"])
        self.assertAlmostEqual(7 / 3, ret["avg_question_len"])

        self.assertEqual(1, ret["min_answer_len"])
        self.assertEqual(3, ret["max_answer_len"])
        self.assertAlmostEqual(2, ret["avg_answer_len"])


class TestPassageStatistics(unittest.TestCase):

    def test_empty(self):
        engine = PassageStatsCalculator()
        ret = engine.calc_statistics([])
        self.assertEqual(0, ret["num_passages"])
        self.assertIsNone(ret["avg_passage_chars"])
        self.assertIsNone(ret["avg_passage_lines"])

    def test_oneline_passage(self):
        engine = PassageStatsCalculator()
        ret = engine.calc_statistics([DataItem("a passage", "q?")])
        self.assertEqual(1, ret["num_passages"])
        self.assertEqual(9, ret["min_passage_chars"], msg=f"actual = {ret}")
        self.assertEqual(9, ret["max_passage_chars"])
        self.assertAlmostEqual(9, ret["avg_passage_chars"])

        self.assertEqual(1, ret["min_passage_lines"])
        self.assertEqual(1, ret["max_passage_lines"])
        self.assertAlmostEqual(1, ret["avg_passage_lines"])

    def test_multiline_passage(self):
        engine = PassageStatsCalculator()
        ret = engine.calc_statistics([DataItem("line1\nline22\nline3\n", "q?")])
        self.assertEqual(1, ret["num_passages"])

        self.assertEqual(3, ret["min_passage_lines"])
        self.assertEqual(3, ret["max_passage_lines"])
        self.assertAlmostEqual(3, ret["avg_passage_lines"])

    def test_multiline_passage(self):
        engine = PassageStatsCalculator()
        ret = engine.calc_statistics([
            DataItem("line1\nline2\nline33\n", "q?"),
            DataItem("line11\nline22", "q?"),
            DataItem(None, "q?"),
        ])
        self.assertEqual(2, ret["num_passages"])

        self.assertEqual(13, ret["min_passage_chars"])
        self.assertEqual(19, ret["max_passage_chars"])

        self.assertEqual(2, ret["min_passage_lines"])
        self.assertEqual(3, ret["max_passage_lines"])
        self.assertAlmostEqual(2.5, ret["avg_passage_lines"], msg=f"actual = {ret}")

    def test_unique_passages(self):
        engine = PassageStatsCalculator()
        ret = engine.calc_statistics([
            DataItem("ups duplicated", "q?"),
            DataItem("ups duplicated", "q?"),
            DataItem("p3", "q?"),
        ])
        self.assertEqual(3, ret["num_passages"])
        self.assertEqual(2, ret["num_unique_passages"])
