import unittest
from io import StringIO

from src.datafile_parser import DataItem, DatafileParser
from src.get_root import get_root


class TestDatafileParser(unittest.TestCase):

    def test_parse_id(self):
        engine = DatafileParser()
        engine.process_line("# newdoc id = f-6VWP66LZ\n", {}, 0)
        self.assertEqual("f-6VWP66LZ", engine.current_id)

    def test_parse_question(self):
        engine = DatafileParser()
        engine.process_line("# question 1-0 = What should be served?\n", {}, 12)
        self.assertEqual(["What should be served?"], engine.current_questions)

    def test_parse_answer(self):
        engine = DatafileParser()
        engine.process_line("# answer 18-3 = N/A\n", {}, 14)
        self.assertEqual(["N/A"], engine.current_answers)

    def test_parse_texts(self):
        engine = DatafileParser()
        engine.process_line("# text = Preheat the oven to 350deg F.\n", {}, 14)
        engine.process_line("# text = Heat the olive oil and butter in a large skillet until the butter.\n", {}, 15)

        self.assertEqual(["Preheat the oven to 350deg F.",
                          "Heat the olive oil and butter in a large skillet until the butter."], engine.current_texts)

    def test_close_item(self):
        engine = DatafileParser()  # note internal test
        engine.current_id = "123"
        engine.current_questions = ["q1", "q2"]
        engine.current_answers = ["a1", "a2"]
        engine.current_texts = ["line1", "line2", "line3"]
        engine.current_subids = ["s1", "s2"]

        ret = engine.close_current_data_item()
        self.assertIsInstance(ret, list)

        first = ret[0]
        self.assertIsInstance(first, DataItem)
        self.assertEqual("123", first.id)
        self.assertEqual("q1", first.question)
        self.assertEqual("a1", first.answer)
        self.assertEqual("s1", first.subid)
        self.assertEqual("line1\nline2\nline3", first.passage)

        second = ret[1]
        self.assertIsInstance(second, DataItem)
        self.assertEqual("123", second.id)
        self.assertEqual("q2", second.question)
        self.assertEqual("a2", second.answer)
        self.assertEqual("s2", second.subid)
        self.assertEqual("line1\nline2\nline3", second.passage)

    def test_parse_from_stream(self):
        a_stream = StringIO(
            "# newdoc id = f-GPT7GGG6\n"
            "# question 18-0 = Where do you store the small ball?\n"
            "# answer 18-0 = N/A\n"
            "# question 10-2 = How do you cook pasta in large salted pot of boiling water?\n"
            "# answer 10-2 = cook pasta according to pasta package directions\n"
            "# text = Bake it for 12 minutes.\n"
            "# text = Remove weights and paper and set them aside.\n"
            "# text = In a small skillet cook the bacon until crisp.\n"
        )
        engine = DatafileParser()
        ret = engine.parse_from_stream(a_stream)
        self.assertEqual(2, len(ret))

        first = ret[0]
        self.assertIsInstance(first, DataItem)
        self.assertEqual("f-GPT7GGG6", first.id)
        self.assertEqual("Where do you store the small ball?", first.question)
        self.assertEqual("18-0", first.subid)
        self.assertEqual("N/A", first.answer)

        second = ret[1]
        self.assertIsInstance(second, DataItem)
        self.assertEqual("How do you cook pasta in large salted pot of boiling water?", second.question)
        self.assertEqual("cook pasta according to pasta package directions", second.answer)
        self.assertEqual("10-2", second.subid)
        self.assertRegex(second.passage.replace("\n", "  ## "), "Bake it for 12")
        self.assertRegex(second.passage.replace("\n", "  ## "), "cook the bacon until crisp")

    def test_parse_from_file(self):
        resource = f"{get_root()}/modules/recipe2video/data/train/crl_srl.csv"
        dataset = DatafileParser().parse_from_file(resource, limit=796)
        self.assertEqual(27 + 57, len(dataset))
        first = dataset[0]
        self.assertEqual("f-6VWP66LZ", first.id)
        self.assertEqual("How many actions does it take to process the minced meat?", first.question)
        self.assertEqual("1", first.answer)

        last = dataset[-1]
        self.assertEqual("f-GGX2LSGX", last.id)
        self.assertEqual("What should be baked in the oven?", last.question)
        self.assertEqual("the pie crust", last.answer)

    def test_get_resource_val(self):
        res = DatafileParser.get_resource("val")
        self.assertGreaterEqual(len(res), 3000)
        self.assertIsNotNone(res[0].answer)
        self.assertIsNotNone(res[99].answer)

    def test_get_resource_test(self):
        res = DatafileParser.get_resource("test")
        self.assertGreaterEqual(len(res), 3000)
        # self.assertIsNone(res[0].answer)  # no longer true, the annotated test set was shared in Feb 2022
        # self.assertIsNone(res[99].answer)

    def test_get_bad_resource(self):
        with self.assertRaises(ValueError):
            DatafileParser.get_resource("bad resource")
