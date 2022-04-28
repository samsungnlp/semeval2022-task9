from unittest import TestCase
from src.data_statistics_extractive_answer import ExtractiveAnswerChecker, DataItem, RecoverableAnswerChecker


class TestDataStatisticsExtractiveAnswer(TestCase):

    def test_empty(self):
        engine = ExtractiveAnswerChecker()
        ret = engine.calc_statistics([])
        self.assertEqual(0, ret["num_extractive_and_nonextractive_answers"])
        self.assertEqual(0, ret["num_extractive_answers"])
        self.assertEqual(0, ret["num_non_extractive_answers"])

    def test_extractve_item(self):
        engine = ExtractiveAnswerChecker()
        item = DataItem("Remove skin.\nPlace in a\nbuttered baking dish and sprinkle with salt and pepper.\n"
                        "Brush with remaining"
                        , question="", answer="in a buttered baking dish")
        ret = engine.calc_statistics([item])
        self.assertEqual(1, ret["num_extractive_and_nonextractive_answers"])
        self.assertEqual(1, ret["num_extractive_answers"])
        self.assertEqual(0, ret["num_non_extractive_answers"])

    def test_non_extractve_item(self):
        engine = ExtractiveAnswerChecker()
        item = DataItem("Remove skin.\nPlace in a buttered\nbaking dish and sprinkle with salt and pepper."
                        , question="", answer="the halibut")
        ret = engine.calc_statistics([item])
        self.assertEqual(1, ret["num_extractive_and_nonextractive_answers"])
        self.assertEqual(0, ret["num_extractive_answers"])
        self.assertEqual(1, ret["num_non_extractive_answers"])

    def test_multiple_items(self):
        engine = ExtractiveAnswerChecker()
        item1 = DataItem("A passage 1", question="", answer="a passage")
        item2 = DataItem("A passage 2", question="", answer="N/A")
        item3 = DataItem("A passage 3", question="", answer="passage 3")
        item4 = DataItem("A passage 4", question="", answer="11")
        item5 = DataItem("A passage 5", question="", answer="45")
        item6 = DataItem("A passage 6", question="", answer="cannot be found in passage")
        item7 = DataItem("the first event", question="", answer="the first event")  # should not count to the matched

        ret = engine.calc_statistics([item1, item2, item3, item4, item5, item6, item7])
        self.assertEqual(3, ret["num_extractive_and_nonextractive_answers"])
        self.assertEqual(2, ret["num_extractive_answers"])
        self.assertEqual(1, ret["num_non_extractive_answers"])

    def test_is_extractive(self):
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer(None))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("N/A"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("1"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("2"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("3"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("9"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("10"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("11"))

        self.assertTrue(ExtractiveAnswerChecker.is_textual_answer("pan"))
        self.assertTrue(ExtractiveAnswerChecker.is_textual_answer("1tbs"))

        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("the first event"))
        self.assertFalse(ExtractiveAnswerChecker.is_textual_answer("the second event"))


class TestRecoverableAnswer(TestCase):

    def test_invalid(self):
        engine = RecoverableAnswerChecker()
        ret = engine.calc_statistics([DataItem("numeric", "?", answer="1"),
                                      DataItem("ordering", "?", answer="the first event"),
                                      DataItem("no answer", "?", answer="N/A"),
                                      DataItem("extractive", "?", answer="extractive")])

        self.assertEqual(4, ret["recoverable__skipped_answers"])
        self.assertEqual(0, ret["recoverable__analysed_answers"])
        self.assertEqual(0, ret["recoverable__after_removing_article"])
        self.assertEqual(0, ret["recoverable__after_removing_heading_until"])
        self.assertEqual(0, ret["recoverable__after_removing_heading_with"])
        self.assertEqual(0, ret["recoverable__after_removing_heading_by"])
        self.assertEqual(0, ret["recoverable__non_recoverable"])

    def test_recoverable_article(self):
        engine = RecoverableAnswerChecker()
        ret = engine.calc_statistics([DataItem("this is answer", "?", answer="This is the answer")])

        self.assertEqual(1, ret["recoverable__analysed_answers"], msg=f"actual = {ret}")
        self.assertEqual(1, ret["recoverable__after_removing_article"], msg=f"actual = {ret}")
        self.assertEqual(0, ret["recoverable__non_recoverable"])

    def test_recoverable_until(self):
        engine = RecoverableAnswerChecker()
        ret = engine.calc_statistics([DataItem("Still it melts", "?", answer="until it melts")])

        self.assertEqual(1, ret["recoverable__analysed_answers"], msg=f"actual = {ret}")
        self.assertEqual(1, ret["recoverable__after_removing_heading_until"], msg=f"actual = {ret}")
        self.assertEqual(0, ret["recoverable__non_recoverable"])

    def test_recoverable_by(self):
        engine = RecoverableAnswerChecker()
        ret = engine.calc_statistics([DataItem("with a spoon", "?", answer="by spoon")])

        self.assertEqual(1, ret["recoverable__analysed_answers"], msg=f"actual = {ret}")
        self.assertEqual(1, ret["recoverable__after_removing_heading_by"], msg=f"actual = {ret}")
        self.assertEqual(0, ret["recoverable__non_recoverable"])

    def test_non_recoverable(self):
        engine = RecoverableAnswerChecker()
        ret = engine.calc_statistics([DataItem("this is IDK what", "?", answer="completely meaningless")])

        self.assertEqual(1, ret["recoverable__analysed_answers"], msg=f"actual = {ret}")
        self.assertEqual(0, ret["recoverable__after_removing_article"], msg=f"actual = {ret}")
        self.assertEqual(0, ret["recoverable__after_removing_heading_by"], msg=f"actual = {ret}")
        self.assertEqual(1, ret["recoverable__non_recoverable"])
