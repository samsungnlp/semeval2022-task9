from src.pipeline.handler_metrics import HandlerF1, QuestionAnswerRecipe, PredictedAnswer, HandlerExactMatch
from src.unpack_data import Q_A
from io import StringIO
import unittest


class TestHandlerMetrics(unittest.TestCase):

    def test_no_answers_provided(self):
        questions = [QuestionAnswerRecipe(Q_A("# question 1-2 = q?"), recipe=None)]
        answers = [PredictedAnswer("dummy")]
        engine = HandlerF1(StringIO())
        engine.handle_questions_answers(questions, answers)
        self.assertEqual("N/A", engine.last_stored_f1)

    def test_should_accept_empty_lists(self):
        sink = StringIO()
        engine = HandlerF1(sink)
        engine.handle_questions_answers([], [])
        self.assertEqual("N/A", engine.last_stored_f1)
        dumped_logs = sink.getvalue().replace("\n", " ## ")
        self.assertRegex(dumped_logs, "[Ff]1 = N/A")

    def test_full_match(self):
        questions = [QuestionAnswerRecipe(Q_A.build_dummy_qa("q?", "1", "this is an answer"), recipe=None)]
        answers = [PredictedAnswer("this is an answer")]
        sink = StringIO()
        engine = HandlerF1(sink)
        engine.handle_questions_answers(questions, answers)
        self.assertAlmostEqual(1.0, engine.last_stored_f1)
        dumped_logs = sink.getvalue().replace("\n", " ## ")
        self.assertRegex(dumped_logs, "[Ff]1 = 1\\.0")

    def test_partial_match(self):
        questions = [QuestionAnswerRecipe(Q_A.build_dummy_qa("?", "1", "this is weird"), recipe=None)]
        answers = [PredictedAnswer("this is answer")]
        engine = HandlerF1(StringIO())
        engine.handle_questions_answers(questions, answers)
        self.assertTrue(.5 <= engine.last_stored_f1 <= .8)

    def test_avg_of_f1s(self):
        questions = [QuestionAnswerRecipe(Q_A.build_dummy_qa("q?", "0", "full match"), recipe=None),
                     QuestionAnswerRecipe(Q_A.build_dummy_qa("q?", "1", "complete mismatch"), recipe=None)]
        answers = [PredictedAnswer("full match"),
                   PredictedAnswer("utter failure")]
        engine = HandlerF1(StringIO())
        engine.handle_questions_answers(questions, answers)
        self.assertAlmostEqual(.5, engine.last_stored_f1)

    def test_duplicated_token_problem(self):
        answer = "in a sealed container in a cool dark place"
        qs = [QuestionAnswerRecipe(Q_A.build_dummy_qa("q?", "1", answer), recipe=None)]
        answers = [PredictedAnswer(answer)]
        engine = HandlerF1(StringIO())
        engine.handle_questions_answers(qs, answers)
        self.assertAlmostEqual(1.0, engine.last_stored_f1)


class TestHandlerExactMatch(unittest.TestCase):
    def test_empty_list(self):
        engine = HandlerExactMatch()
        engine.handle_questions_answers([], [])
        self.assertEqual("N/A", engine.last_result)

    def test_no_answer_provided(self):
        sink = StringIO()
        engine = HandlerExactMatch(sink)
        engine.handle_questions_answers([QuestionAnswerRecipe(Q_A.build_dummy_qa("?", "1", None), recipe=None)],
                                        [PredictedAnswer("whatever")])
        self.assertEqual("N/A", engine.last_result)
        dumped_logs = sink.getvalue().replace("\n", " ## ")
        self.assertRegex(dumped_logs, "Exact [mM]atch = N/A")

    def test_full_match(self):
        sink = StringIO()
        engine = HandlerExactMatch(sink)
        engine.handle_questions_answers([QuestionAnswerRecipe(Q_A.build_dummy_qa("?", "1", "this is answer"), None)],
                                        [PredictedAnswer("this is  answer")])
        self.assertAlmostEqual(1.0, engine.last_result)
        dumped_logs = sink.getvalue().replace("\n", " ## ")
        self.assertRegex(dumped_logs, "Exact [mM]atch = 1\\.0")

    def test_avg_of_matches(self):
        engine = HandlerExactMatch(StringIO())
        questions = [QuestionAnswerRecipe(Q_A.build_dummy_qa("?", "1", "good match 1"), None),
                     QuestionAnswerRecipe(Q_A.build_dummy_qa("?", "2", "bad match"), None),
                     QuestionAnswerRecipe(Q_A.build_dummy_qa("?", "3", "good match 2"), None)]
        answers = [PredictedAnswer("good match 1"),
                   PredictedAnswer("ups, bad match"),
                   PredictedAnswer("good match 2 ")]
        engine.handle_questions_answers(questions, answers)
        self.assertAlmostEqual(2.0 / 3, engine.last_result)
