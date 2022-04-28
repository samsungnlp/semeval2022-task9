from src.unpack_data import QuestionAnswerRecipe
from src.pipeline.question_category import QuestionCategory
from typing import Any, Dict, List, Optional
import abc


class PredictedAnswer:

    def __init__(self, answer: Optional[str], raw_question: str = None, confidence: float = None,
                 more_info: Dict[str, Any] = {}):
        """
        :param answer: answer to the question / None if the engine cannot answer it / N/A for no answer in the passage
        :param raw_question: Raw question (please copy the question)
        :param confidence: confidence in [0,1] or None if not supported.
        :param more_info: Additional info to be passed back to the user
        """
        if answer == "":
            raise ValueError("Answer cannot be empty string. "
                             "Provide N/A for 'no answer in passage' or None if the engine cannot provide it")

        if confidence is not None and not (0 <= confidence <= 1):
            raise ValueError("Incorrect value for confidence: expecting confidence in [0,1] or None. Actual value"
                             f" = {confidence}")

        self.answer = answer
        self.raw_question = raw_question
        self.confidence = confidence
        self.more_info = more_info

    def has_answer(self) -> bool:
        """
        :return: True if the question was answered. Note. We consider "N/A" as a valid answer!
        """
        return self.answer is not None


class InterfaceQuestionAnswering(abc.ABC):
    """
    Basic interface class
    """

    @abc.abstractmethod
    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:  # pragma: nocover
        """
        :param question: question to be answered.
        :param question_category: Predicted category of the question
        :param more_info: Additional info to be passed to the engine. Implementation-dependent.
        :return: Predicted answer. The engine is allowed to return PredictedAnswer(None) as an indicator,
          that it doesn't know how to handle it
        """
        raise NotImplementedError("I must be implemented in a derived class")

    @abc.abstractmethod
    def batch_answer_questions(self, questions: List[QuestionAnswerRecipe], categories: List[QuestionCategory],
                               more_info: Dict[str, Any] = {}) -> List[PredictedAnswer]:  # pragma: nocover
        """
        can be overridden in a derived class eg. for BERT batch prediction
        :param questions: questions to be answered
        :param categories: predicted categories, one per question
        :param more_info: additional info to be passed to the classifiers
        :return: List of answers, the engine is allowed to return N
        """
        raise NotImplementedError("I must be implemented in a derived class")


class QuestionAnsweringBase(InterfaceQuestionAnswering):

    def batch_answer_questions(self, questions: List[QuestionAnswerRecipe], categories: List[QuestionCategory],
                               more_info: Dict[str, Any] = {}) -> List[PredictedAnswer]:
        """
        stupid iterative implementation. Should be done differently for BERT batch calls
        :param questions: questions to be answered
        :param categories: predicted categories, one per question
        :param more_info: additional info to be passed to the classifiers
        :return: List of answers, the engine is allowed to return N
        """
        if len(questions) != len(categories):
            raise ValueError()

        # warning! virtual call goes here!
        return [self.answer_a_question(q, c) for q, c in zip(questions, categories)]

    @abc.abstractmethod
    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:  # pragma: nocover
        """
        :param question: question to be answered.
        :param question_category: Predicted category of the question
        :param more_info: Additional info to be passed to the engine. Implementation-dependent.
        :return: Predicted answer. The engine is allowed to return PredictedAnswer(None) as an indicator,
          that it doesn't know how to handle it
        """
        raise NotImplementedError("I must be implemented in a derived class")
