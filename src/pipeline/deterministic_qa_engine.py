from typing import Dict, Any

from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory


class QuestionAnswererNA(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer N/A"

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: ignored
        :param more_info: ignored
        :return: "N/A" regardless of the question
        """
        more_info_for_answer = {"source": QuestionAnswererNA.DESCRIPTION}
        return PredictedAnswer(None, raw_question=question.question, confidence=None, more_info=more_info_for_answer)


class QuestionAnswererConstantAnswer(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer ConstantAnswer"

    def __init__(self, answer_to_be_returned: str):
        if answer_to_be_returned == "":
            raise ValueError("Answer cannot be an empty string")

        self.answer_to_be_returned = answer_to_be_returned

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: ignored
        :param more_info: ignored
        :return: predefined answer regardless of the question
        """
        more_info_for_answer = {"source": QuestionAnswererConstantAnswer.DESCRIPTION}
        return PredictedAnswer(self.answer_to_be_returned, raw_question=question.question, confidence=None
                               , more_info=more_info_for_answer)
