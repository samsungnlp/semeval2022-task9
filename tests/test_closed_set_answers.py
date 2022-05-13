from src.data_statistics_closed_set_answers import ClosedSetAnswerChecker, DataItem
import unittest


class TestClosedSetAnswerChecker(unittest.TestCase):

    def test_not_closed_answers(self):
        ret = ClosedSetAnswerChecker().calc_statistics([DataItem("", "", answer="bad answer")])
        self.assertEqual(0, ret["closed_answers_by_using_a_knife"])
        self.assertEqual(1, ret["closed_answers_remaining_answers"])

    def test_not_closed_answers(self):
        ret = ClosedSetAnswerChecker().calc_statistics([DataItem("", "", answer="by using a knife"),
                                                        DataItem("", "", answer="bowl")])
        self.assertEqual(1, ret["closed_answers_by_using_a_knife"])
        self.assertEqual(1, ret["closed_answers_bowl"])
        self.assertEqual(0, ret["closed_answers_the_first_event"])
        self.assertEqual(0, ret["closed_answers_remaining_answers"])
