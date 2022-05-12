import unittest

from src.pipeline.question_answering_dispatcher import QuestionAnsweringDispatcher, PredictedAnswer
from src.pipeline.deterministic_qa_engine import QuestionAnswererNA, QuestionAnswererConstantAnswer
from src.unpack_data import QuestionAnswerRecipe, Q_A


class TestQuestionAnsweringDispatcher(unittest.TestCase):

    def test_default_dispatching_rules(self):
        engine = QuestionAnsweringDispatcher()

        question = QuestionAnswerRecipe(Q_A("# question A-B = Q?"), None)
        ret = engine.predict_answer(question)
        self.assertIsInstance(ret, PredictedAnswer)
        self.assertFalse(ret.has_answer())
        self.assertEqual("not_recognized", ret.more_info["predicted_category"])
        self.assertEqual("QuestionAnswerer N/A", ret.more_info["source"])

    def test_custom_dispatching_rules(self):
        dispatching_rules = {
            "counting_actions": QuestionAnswererConstantAnswer("1"),
            "event_ordering": QuestionAnswererConstantAnswer("the first event"),
            "location_srl": QuestionAnswererNA()
        }
        engine = QuestionAnsweringDispatcher(dispatching_rules)

        questions = [
            QuestionAnswerRecipe(Q_A("# question A-B = Where do you place bean sprouts?"), None),
            QuestionAnswerRecipe(Q_A("# question A-B = How many actions does it take to process the minced "
                                     "meat?"), None),
            QuestionAnswerRecipe(Q_A("# question A-B = Cutting the stem into bite - size pieces into bite - size"
                                     " pieces and sauting minced meat in a separate pan, which comes first?"), None)
        ]

        ret = engine.predict_answers('test', False, questions)
        self.assertIsInstance(ret, list)
        self.assertFalse(ret[0].has_answer())
        self.assertTrue(ret[1].has_answer())
        self.assertEqual("1", ret[1].answer)
        self.assertTrue(ret[2].has_answer())
        self.assertEqual("the first event", ret[2].answer)
