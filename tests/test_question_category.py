import unittest

from src.pipeline.question_category import QuestionCategory, GetCategoryFromQuestionId, GetCategoryFromQuestionStructure
from src.unpack_data import QuestionAnswerRecipe, Recipe, Q_A


class TestQuestionCategory(unittest.TestCase):

    def test_determine_description(self):
        qc = QuestionCategory("0_times")
        self.assertIn("Counting times? A: Number", qc.description)

    def test_determine_description_str(self):
        qc = QuestionCategory("4")
        self.assertIn("X, Y which comes first?", qc.description)

    def test_no_description(self):
        qc = QuestionCategory("whatever")
        self.assertEqual("N/A", qc.description)


class TestGetCategoryFromQuestionId(unittest.TestCase):

    def test_stupid_classifier(self):
        engine = GetCategoryFromQuestionId()
        recipe = Recipe(["# newdoc id = fake", "# newpar id = fake::ingredients"])
        dummy_question = QuestionAnswerRecipe(qa=Q_A("# question 3-4 = q?", "answer = a"), recipe=recipe)
        a_class = engine.predict_category(dummy_question)
        self.assertIsNotNone(a_class)
        self.assertEqual("3_how", a_class.category)

    def test_no_category(self):
        engine = GetCategoryFromQuestionId()
        recipe = Recipe(["# newdoc id = fake", "# newpar id = fake::ingredients"])
        dummy_question = QuestionAnswerRecipe(qa=Q_A("# question 99-2 = q?", "answer = a"), recipe=recipe)
        a_class = engine.predict_category(dummy_question)
        self.assertIsNone(a_class)


class TestGetCategoryFromQuestionStructure(unittest.TestCase):

    def test_regex_classifier_class_0_times(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 0-0 = How many times is the bowl used?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("0_times", a_class.category)

    def test_regex_classifier_class_0_actions(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 0-0 = How many actions does it take to process the tomato?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("0_actions", a_class.category)

    def test_regex_classifier_class_0_are_used(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 0-1 = How many spoons are used?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("0_are_used", a_class.category)

    def test_regex_classifier_class_1(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 1-0 = What should be served?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("1", a_class.category)

    def test_regex_classifier_class_2_where(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 2-2 = Where should you add the chopped vegetables?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("2_where", a_class.category)

    def test_regex_classifier_class_2_how(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 2-2 = How do you brush the salad dressing?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("2_6_10_14", a_class.category)

    def test_regex_classifier_class_3_How(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 3-3 = How did you get the cooked vegetable?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("3_how", a_class.category)

    def test_regex_classifier_class_3_What(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 3-0 = What's in the lentil salad?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("3_what", a_class.category)

    def test_regex_classifier_class_4(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A(
            "# question 4-5 = Cutting the stem into bite - size pieces into bite - size pieces and sauting minced meat in a separate pan, which comes first?",
            "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("4", a_class.category)

    def test_regex_classifier_class_5(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 5-0 = To what extent do you cut carrots and zucchini?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("5", a_class.category)

    def test_regex_classifier_class_6(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 6-1 = How do you prick the dough slightly?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("2_6_10_14", a_class.category)

    def test_regex_classifier_class_7(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 7-1 = For how long do you boil the potatoes until cooked?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("7", a_class.category)

    def test_regex_classifier_class_8(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 8-2 = Where do you season the trout with salt and pepper?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("8", a_class.category)

    def test_regex_classifier_class_9(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 9-0 = By how much do you cover the beans with water in a pot?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("9", a_class.category)

    def test_regex_classifier_class_10(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 10-1 = How do you coat hot syrup mixture the popcorn nut mixture?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("2_6_10_14", a_class.category)

    def test_regex_classifier_class_11(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 11-0 = Why do you use gas?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("11_15", a_class.category)

    def test_regex_classifier_class_12_what(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 12-1 = What do you mix the oil in a small bowl with?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("12_13", a_class.category)

    def test_regex_classifier_class_13(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 13-0 = What do you put the raspberries into a liqudizer with?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("12_13", a_class.category)

    def test_regex_classifier_class_14(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 14-0 = How do you use the same pot of water??"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("2_6_10_14", a_class.category)

    def test_regex_classifier_class_15(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 15-1 = Why do you pinch the pizza dough?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("11_15", a_class.category)

    def test_regex_classifier_class_16(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 16-0 = From where do you remove the spinach and shallots mix?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("16", a_class.category)

    def test_regex_classifier_class_17(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 17-0 = Where was the stuffed mushroom before it was garnished?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("17", a_class.category)

    def test_regex_classifier_class_18(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 18-1 = To what extent do you cut the shortening in?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("5", a_class.category)

    def test_regex_classifier_class_14_preheat(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(Q_A("# question 14-3 = How do you preheat your oven?"), None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("14_preheat", a_class.category)

    def test_regex_classifier_class_14_preheat_alt_spelling(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(Q_A("# question 14-0 = How do you pre - heat the oven?", "answer = a"), None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("14_preheat", a_class.category)
