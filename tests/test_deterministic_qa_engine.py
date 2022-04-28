from src.pipeline.interface_question_answering import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe, Q_A
from src.pipeline.deterministic_qa_engine import QuestionAnswererNA, QuestionAnswererConstantAnswer

import unittest


class TestNaEngine(unittest.TestCase):

    def test_na_answer(self):
        engine = QuestionAnswererNA()

        question = QuestionAnswerRecipe(Q_A("# question 1-2 = What is this?"), recipe=None)
        res = engine.answer_a_question(question, QuestionCategory("1"))
        self.assertIsNone(res.answer)
        self.assertFalse(res.has_answer())
        self.assertIn("source", res.more_info)

    def test_batch_answer_different_list_lengths(self):
        questions = [QuestionAnswerRecipe(Q_A("# question 1-2 = What is this?"), recipe=None)]
        categories = [QuestionCategory('1'), QuestionCategory('2_6_10_14')]

        engine = QuestionAnswererNA()
        with self.assertRaises(ValueError):
            engine.batch_answer_questions(questions, categories)

    def test_batch_answers(self):
        questions = [QuestionAnswerRecipe(Q_A("# question 1-2 = What is this?"), recipe=None),
                     QuestionAnswerRecipe(Q_A("# question 2-3 = This is what?"), recipe=None)]
        categories = [QuestionCategory('1'), QuestionCategory('2_6_10_14')]

        engine = QuestionAnswererNA()
        res = engine.batch_answer_questions(questions, categories)
        self.assertEqual(2, len(res))
        self.assertFalse(res[0].has_answer())
        self.assertFalse(res[1].has_answer())


class TestConstAnswerEngine(unittest.TestCase):

    def test_answer_1(self):
        engine = QuestionAnswererConstantAnswer("1")

        question = QuestionAnswerRecipe(Q_A("# question 2-3 = How much?"), recipe=None)
        res = engine.answer_a_question(question, QuestionCategory('0'))
        self.assertTrue(res.has_answer())
        self.assertEqual("1", res.answer)
        self.assertIn("source", res.more_info)

    def test_answer_the_first_event(self):
        engine = QuestionAnswererConstantAnswer("the first event")

        question = QuestionAnswerRecipe(Q_A("# question 3-4 = A, B which comes first?"), recipe=None)
        res = engine.answer_a_question(question, QuestionCategory('4'))
        self.assertTrue(res.has_answer())
        self.assertEqual("the first event", res.answer)

    def test_bad_answer(self):
        with self.assertRaises(ValueError):
            QuestionAnswererConstantAnswer("")
