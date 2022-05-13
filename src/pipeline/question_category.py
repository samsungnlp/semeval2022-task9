import abc
import re
from typing import Dict, Any, Optional

from src.unpack_data import QuestionAnswerRecipe


class QuestionCategory:
    # The order in the dictionary matters
    CATEGORIES = [
        "counting_times", "counting_actions", "counting_uses", "ellipsis", "location_crl",
        "method_preheat",  # Note!!! this category must be placed before method
        "method", "lifespan_how", "lifespan_what", "event_ordering", "result", "time", "copatient",
        "location_srl", "extent", "purpose", "source", "location_change", "not_recognized", "RC"
    ]

    DEFAULT_DESCRIPTIONS = {
        "counting_times": "Counting times? A: Number",
        "counting_actions": "Counting actions? A: Number",
        "counting_uses": "Counting used? A: Number",
        "ellipsis": "What should be + participle? A: Patient",
        "method_preheat": "How do you preheat? A: Goal",
        "location_crl": "Where should you A: Habitat",
        "method": "How do you? A: Method",
        "lifespan_how": "How did you get? A: Patient",
        "lifespan_what": "What is in? A: Patient",
        "event_ordering": "X, Y which comes first? A: the first|second event",
        "result": "To what extent? A: Result",
        "time": "For how long? A: Time",
        "location_srl": "Where do you? A: Location/Destination",
        "extent": "By how much? A: Extent",
        "purpose": "Why? A: Cause/Purpose",
        "copatient": "What do you? A: Co-Patient/Co-Theme",
        "source": "From where? A: Source",
        "location_change": "Where was X before? A: Habitat",
        "not_recognized": "Not recognized? A: ???",
    }

    CATEGORY_REGEX = {
        "counting_times": r"^How many times.",
        "counting_actions": r"^How many actions.",
        "counting_uses": r"^How many .*are used\?",
        "ellipsis": r"^What should.",
        "location_crl": r"^Where should you.",
        "method": r"^How do you.",
        "lifespan_how": r"^How did you get.",
        "lifespan_what": r"^What's in the.",
        "event_ordering": r".which comes first\?",
        "result": r"^To what exten.",
        "time": r"^For how long.",
        "location_srl": r"^Where do you.",
        "extent": r"^By how much.",
        "purpose": "^Why do you.",
        "copatient": "^What do you.",
        "method_preheat": r"^How do you (re|pre)?.*heat .*(oven|grill)\?$",
        "source": r"^From where.",
        "location_change": r"^Where was.",
        "not_recognized": r".",
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


class GetCategoryFromQuestionStructure(QuestionCategoryClassifier):

    def predict_category(self, question: QuestionAnswerRecipe) -> Optional[QuestionCategory]:
        for category in QuestionCategory.CATEGORIES:
            if re.search(QuestionCategory.CATEGORY_REGEX[category], question.question):
                return QuestionCategory(category)

        return QuestionCategory("not_recognized")
