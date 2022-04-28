import unittest

from src.data_question_class_statistics import QuestionClassStatistics, DataItem
import tempfile
import pathlib


class TestQuestionClassStatistics(unittest.TestCase):

    def test_bad_id(self):
        engine = QuestionClassStatistics()
        with self.assertRaises(ValueError):
            engine.process_item(DataItem("", "", sub_id="bad_subid_format"))

    def test_no_items(self):
        engine = QuestionClassStatistics()
        res = engine.calc_statistics([])
        self.assertIsInstance(res, dict)
        self.assertEqual(19, len(res))
        for k, v in res.items():
            self.assertRegex(k, "question_class_[0-9]*")
            self.assertEqual(0, v)

    def test_single_item(self):
        engine = QuestionClassStatistics()
        res = engine.calc_statistics([DataItem("p", "q", sub_id="1-2")])
        self.assertIsInstance(res, dict)
        self.assertEqual(19, len(res))

        self.assertEqual(1, res["question_class_1"])
        self.assertEqual(0, res["question_class_0"])

        for x in range(2, 19):
            self.assertEqual(0, res[f"question_class_{x}"])

    def test_multiple_items(self):
        engine = QuestionClassStatistics()
        res = engine.calc_statistics([
            DataItem("p1", "q1", sub_id="1-2"),
            DataItem("p2", "q2", sub_id="1-4"),
            DataItem("p3", "q3", sub_id="18-5")])
        self.assertIsInstance(res, dict)
        self.assertEqual(19, len(res))

        self.assertEqual(0, res["question_class_0"])
        self.assertEqual(2, res["question_class_1"])
        self.assertEqual(1, res["question_class_18"])

        for x in range(2, 18):
            self.assertEqual(0, res[f"question_class_{x}"])

    def test_save_to_excel(self):
        engine = QuestionClassStatistics()
        for i in range(0, 19):
            engine.process_item(DataItem("p", "q", answer="a", id="id", sub_id=f"{i}-0"))

        with tempfile.TemporaryDirectory() as dir:
            engine.save_to_excel_workbook_per_class(prefix=dir)
            self.assertTrue(pathlib.Path(f"{dir}/question_category_0.xls").exists())
            self.assertTrue(pathlib.Path(f"{dir}/question_category_18.xls").exists())
