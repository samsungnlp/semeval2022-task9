import unittest

from src.pipeline.question_category import QuestionCategory, GetCategoryFromQuestionStructure
from src.unpack_data import QuestionAnswerRecipe, Recipe, Q_A


class TestQuestionCategory(unittest.TestCase):

    def test_determine_description(self):
        qc = QuestionCategory("counting_times")
        self.assertIn("Counting times? A: Number", qc.description)

    def test_determine_description_str(self):
        qc = QuestionCategory("event_ordering")
        self.assertIn("X, Y which comes first?", qc.description)

    def test_no_description(self):
        qc = QuestionCategory("whatever")
        self.assertEqual("N/A", qc.description)


class TestGetCategoryFromQuestionStructure(unittest.TestCase):

    def test_regex_classifier_class_counting_times(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = How many times is the bowl used?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("counting_times", a_class.category)

    def test_regex_classifier_class_counting_actions(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = How many actions does it take to process the tomato?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("counting_actions", a_class.category)

    def test_regex_classifier_class_counting_uses(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = How many spoons are used?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("counting_uses", a_class.category)

    def test_regex_classifier_class_ellipsis(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = What should be served?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("ellipsis", a_class.category)

    def test_regex_classifier_class_location_crl(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = Where should you add the chopped vegetables?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("location_crl", a_class.category)

    def test_regex_classifier_class_how_1(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = How do you brush the salad dressing?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("method", a_class.category)

    def test_regex_classifier_class_how_2(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = How did you get the cooked vegetable?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("lifespan_how", a_class.category)

    def test_regex_classifier_class_lifespan_what(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = What's in the lentil salad?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("lifespan_what", a_class.category)

    def test_regex_classifier_class_event_ordering(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A(
            "# question 20-9 = Cutting the stem into bite - size pieces into bite - size pieces and sauting minced "
            "meat in a separate pan, which comes first?",
            "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("event_ordering", a_class.category)

    def test_regex_classifier_class_result(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = To what extent do you cut carrots and zucchini?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("result", a_class.category)

    def test_regex_classifier_class_how_3(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = How do you prick the dough slightly?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("method", a_class.category)

    def test_regex_classifier_class_time(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = For how long do you boil the potatoes until cooked?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("time", a_class.category)

    def test_regex_classifier_class_location_srl(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = Where do you season the trout with salt and pepper?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("location_srl", a_class.category)

    def test_regex_classifier_class_extent(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = By how much do you cover the beans with water in a pot?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("extent", a_class.category)

    def test_regex_classifier_class_how_4(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = How do you coat hot syrup mixture the popcorn nut mixture?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("method", a_class.category)

    def test_regex_classifier_class_purpose1(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = Why do you use gas?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("purpose", a_class.category)

    def test_regex_classifier_class_copatient1(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = What do you mix the oil in a small bowl with?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("copatient", a_class.category)

    def test_regex_classifier_class_copatient2(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = What do you put the raspberries into a liqudizer with?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("copatient", a_class.category)

    def test_regex_classifier_class_how_5(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = How do you use the same pot of water??"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("method", a_class.category)

    def test_regex_classifier_class_purpose2(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(qa=Q_A("# question 20-9 = Why do you pinch the pizza dough?", "answer = a"),
                                        recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("purpose", a_class.category)

    def test_regex_classifier_class_source(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = From where do you remove the spinach and shallots mix?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("source", a_class.category)

    def test_regex_classifier_class_location_change(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = Where was the stuffed mushroom before it was garnished?", "answer = a"),
            recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("location_change", a_class.category)

    def test_regex_classifier_class_result_na(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(
            qa=Q_A("# question 20-9 = To what extent do you cut the shortening in?", "answer = a"), recipe=None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("result", a_class.category)

    def test_regex_classifier_class_how_preheat_1(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(Q_A("# question 20-9 = How do you preheat your oven?"), None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("method_preheat", a_class.category)

    def test_regex_classifier_class_how_preheat__alt_spelling(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(Q_A("# question 20-9 = How do you pre - heat the oven?", "answer = a"), None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("method_preheat", a_class.category)

    def test_regex_classifier_not_recognized(self):
        engine = GetCategoryFromQuestionStructure()
        question = QuestionAnswerRecipe(Q_A("# question XYZ-ABC = Is this question recognizable?", "answer = No"), None)
        a_class = engine.predict_category(question)
        self.assertIsNotNone(a_class)
        self.assertEqual("not_recognized", a_class.category)
