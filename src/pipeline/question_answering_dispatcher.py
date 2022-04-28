from typing import Any, Dict, List

import tqdm

from src.pipeline.answerers.bert_NA_answer import BertAnswerNA
from src.pipeline.deterministic_qa_engine import QuestionAnswererNA
from src.pipeline.extractive_qa import refine_prediction as refine_prediction_with_RC
from src.pipeline.interface_question_answering import InterfaceQuestionAnswering, PredictedAnswer
from src.pipeline.question_category import QuestionCategory, QuestionCategoryClassifier, \
    GetCategoryFromQuestionStructure
from src.unpack_data import QuestionAnswerRecipe


class QuestionAnsweringDispatcher:

    def __init__(self, dispatching_table: Dict[str, InterfaceQuestionAnswering] = None,
                 question_classifier: QuestionCategoryClassifier = GetCategoryFromQuestionStructure()):
        """
        :param dispatching_table: Optional: dict[ category_id, answering_engine which should handle the rule]
        """
        self.dispatching_table = dispatching_table if dispatching_table \
            else QuestionAnsweringDispatcher.__build_default_dispatcher()
        self.question_category_classifier: QuestionCategoryClassifier = question_classifier

    @staticmethod
    def __build_default_dispatcher() -> Dict[str, InterfaceQuestionAnswering]:
        return {
            x: QuestionAnswererNA() for x in QuestionCategory.CATEGORIES
        }

    def predict_answer(self, question: QuestionAnswerRecipe, more_info: Dict[str, Any] = {},
                       bert_answer_na: BertAnswerNA = None) -> PredictedAnswer:
        """
        :param bert_answer_na: added if postprocessing with bert NA checker
        :param question: question to be answered
        :param more_info: Additional info to be handled (currently ignored)
        :return:
        """
        category = self.question_category_classifier.predict_category(question)
        assert isinstance(category, QuestionCategory)

        engine = self.dispatching_table[category.category]
        ret = engine.answer_a_question(question=question, question_category=category, more_info=more_info)

        if "RC" in self.dispatching_table:
            rc_engine = self.dispatching_table["RC"]
            ret = refine_prediction_with_RC(engine=rc_engine, question=question, category=category,
                                            rule_based_prediction=ret, more_info=more_info)
        if bert_answer_na and ret.answer and category.category == '4':
            ret.answer = bert_answer_na.check_bert_na_answer(ret.answer, question)
            if ret.answer is None:
                ret.more_info["details_for_excel"] += 'Modify in postprocessing BERT NA'

        ret.more_info["predicted_category"] = category.category
        ret.more_info["predicted_category_description"] = category.description
        ret.more_info["with_postprocessing"] = bool(bert_answer_na)
        ret.more_info["answering_engine"] = engine.__class__.__name__
        return ret

    def predict_answers(self, which_dataset: str, with_postprocessing: bool, questions: List[QuestionAnswerRecipe],
                        more_info: Dict[str, Any] = {}) -> List[PredictedAnswer]:
        use_tqdm = more_info.get("use_tqdm", False)
        iterator = tqdm.tqdm(questions, desc="answering") if use_tqdm else questions
        if with_postprocessing:
            bert_na_answer = BertAnswerNA(which_dataset)
            return [self.predict_answer(q, more_info, bert_na_answer) for q in iterator]

        return [self.predict_answer(q, more_info) for q in iterator]
