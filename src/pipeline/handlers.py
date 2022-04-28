import abc
import json
from typing import List, Dict, Any

from src.pipeline.interface_question_answering import PredictedAnswer, QuestionAnswerRecipe


class InterfaceHandler(abc.ABC):
    @abc.abstractmethod
    def handle_questions_answers(self, questions: List[QuestionAnswerRecipe], answers: List[PredictedAnswer],
                                 more_info: Dict[str, Any] = {}):
        """
        :param questions: source questions
        :param answers: predicted answers
        :param more_info: additional info
        :return: None, side effects allowed
        """


class HandlerSaveToJson(InterfaceHandler):

    def __init__(self, filename: str):
        self.filename = filename

    def handle_questions_answers(self, questions: List[QuestionAnswerRecipe], answers: List[PredictedAnswer],
                                 more_info: Dict[str, Any] = {}):
        if len(questions) != len(answers):
            raise ValueError(f"Mismatching questions vs answers = {len(questions)} vs {len(answers)}")

        json_dict = {}
        for q, a in zip(questions, answers):
            answer_text = a.answer if a.has_answer() else None
            question_id = q.question_class
            recipe_id = q.recipe.id

            if recipe_id not in json_dict:
                json_dict[recipe_id] = {}
            json_dict[recipe_id][question_id] = answer_text

        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=1, ensure_ascii=False)
