import json
import os
from typing import Dict, Any

from src.get_root import get_root
from src.pipeline.interface_question_answering import QuestionAnsweringBase, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe


rc_thr = {
    # "lifespan_what": 0.8,  # disabled due to not improving results
    # "result": 0.99,  # disabled due to not improving results
    # "time": 0.97,  # disabled due to not improving results
    "location_srl": 0.98,
    # "copatient": 0.999,  # disabled due to not improving results
    # "source": 0.0  # disabled due to not improving results
}


def refine_prediction(engine, question: QuestionAnswerRecipe,
                      category: QuestionCategory,
                      rule_based_prediction: PredictedAnswer,
                      more_info: Dict[str, Any] = {}) -> PredictedAnswer:

    rc_pred = engine.answer_a_question(question=question, question_category=category, more_info=more_info)

    if not rc_pred.confidence or rule_based_prediction.answer is not None:
        return rule_based_prediction

    if category.category in rc_thr and rc_pred.confidence >= rc_thr[category.category]:
        rc_pred.more_info["details_for_excel"] = "Added by RC"

        return rc_pred

    return rule_based_prediction


class ExtractiveQuestionAnswerer(QuestionAnsweringBase):
    DESCRIPTION = "Extractive QuestionAnswerer"

    def __init__(self, which_dataset: str = "", predictions_path: str = "data/model_predictions_val_set.json"):
        if which_dataset:
            predictions_path = f"data/model_predictions_{which_dataset}_set.json"

        with open(os.path.join(get_root(), predictions_path), "r") as f:
            self.all_predictions = json.load(f)

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:

        more_info_for_answer = {"source": ExtractiveQuestionAnswerer.DESCRIPTION}

        qa_id = f"{question.recipe.id}-{question.question_class}"
        best_prediction = self.all_predictions[qa_id][0]

        return PredictedAnswer(
            answer=best_prediction["text"],
            raw_question=question.question,
            confidence=best_prediction["probability"],
            more_info=more_info_for_answer
        )


class ExtractiveQuestionAnswererFactory:

    set_type = None

    @staticmethod
    def set_default_engine(set_type: str) -> None:
        assert set_type in ["train", "val", "test"]
        ExtractiveQuestionAnswererFactory.set_type = set_type

    @staticmethod
    def get_extractive_answerer() -> ExtractiveQuestionAnswerer:
        return ExtractiveQuestionAnswerer(ExtractiveQuestionAnswererFactory.set_type)
