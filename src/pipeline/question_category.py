import abc
import re
from typing import Dict, Any, Optional

from src.unpack_data import QuestionAnswerRecipe


class QuestionCategory:
    # The order in the dictionary matters
    CATEGORIES = [
        "0_times", "0_actions", "0_are_used", "1", "2_where", "14_preheat",  # Note!!! this category must be placed before 2_6_10_14
        "2_6_10_14",  "3_how", "3_what", "4", "5", "7", "12_13", "8", "9", "11_15", "16", "17", "18", "RC"
    ]

    DEFAULT_DESCRIPTIONS = {
        "0_times": "Counting times? A: Number",
        "0_actions": "Counting actions? A: Number",
        "0_are_used": "Counting used? A: Number",
        "1": "What should be + participle? A: Patient",
        "14_preheat": "How do you preheat? A: Goal",
        "2_where": "Where should you A: Habitat",
        "2_6_10_14": "How do you? A: Method",
        "3_how": "How did you get? A: Patient",
        "3_what": "What is in? A: Patient",
        "4": "X, Y which comes first? A: the first|second event",
        "5": "To what extent? A: Result",
        "7": "For how long? A: Time",
        "8": "Where do you? A: Location/Destination",
        "9": "By how much? A: Extent",
        "11_15": "Why? A: Cause/Purpose",
        "12_13": "What do you? A: Co-Patient/Co-Theme",
        "16": "From where? A: Source",
        "17": "Where was X before? A: Habitat",
        "18": "Unanswerable? A: None",
    }

    CATEGORY_REGEX = {
        "0_times": r"^How many times.",
        "0_actions": r"^How many actions.",
        "0_are_used": r"^How many .*are used\?",
        "1": r"^What should.",
        "2_where": r"^Where should you.",
        "2_6_10_14": r"^How do you.",
        "3_how": r"^How did you get.",
        "3_what": r"^What's in the.",
        "4": r".which comes first\?",
        "5": r"^To what exten.",
        "7": r"^For how long.",
        "8": r"^Where do you.",
        "9": r"^By how much.",
        "11_15": "^Why do you.",
        "12_13": "^What do you.",
        "14_preheat": r"^How do you (re|pre)?.*heat .*(oven|grill)\?$",
        "16": r"^From where.",
        "17": r"^Where was.",
        "18": r".",
    }

    CLASS_NR_TO_NEW_CLASS = {
        "0": "0",
        "1": "1",
        "2": "2_where",
        "3": "3_how",
        "4": "4",
        "5": "5",
        "6": "2_6_10_14",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "2_6_10_14",
        "11": "11_15",
        "12": "12_13",
        "13": "12_13",
        "14": "2_6_10_14",
        "15": "11_15",
        "16": "16",
        "17": "17",
        "18": "18",
    }

    NO_DESCRIPTION = "N/A"

    def __init__(self, category_id: str, description: str = None):
        self.category = category_id
        self.description = description if description \
            else QuestionCategory.DEFAULT_DESCRIPTIONS.get(self.category, self.NO_DESCRIPTION)
        self.more_info: Dict[str, Any] = {}


class QuestionCategoryClassifier:

    @abc.abstractmethod
    def predict_category(self, question: QuestionAnswerRecipe) -> Optional[QuestionCategory]:
        """
        :param question: question whose category should be predicted
        :return: Category or None if cannot be inferred
        """
        raise NotImplementedError("I must be implemented in a derived class")


class GetCategoryFromQuestionId(QuestionCategoryClassifier):

    def predict_category(self, question: QuestionAnswerRecipe) -> Optional[QuestionCategory]:
        cat = question.question_class.split("-")[0]
        if not cat or cat not in QuestionCategory.CLASS_NR_TO_NEW_CLASS:
            return None

        try:
            ret = QuestionCategory(QuestionCategory.CLASS_NR_TO_NEW_CLASS[cat])
            ret.more_info["source"] = "copied from question id"
            return ret
        except Exception:  # pragma: nocover
            return None


class GetCategoryFromQuestionStructure(QuestionCategoryClassifier):

    def predict_category(self, question: QuestionAnswerRecipe) -> Optional[QuestionCategory]:
        for category in QuestionCategory.CATEGORIES:
            if re.search(QuestionCategory.CATEGORY_REGEX[category], question.question):
                return QuestionCategory(category)

        return QuestionCategory("18")
