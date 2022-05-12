import os
import pathlib
import sys
from typing import Any, Dict, List, TextIO

import pandas

from src.get_root import get_root
from src.pipeline.handler_metrics import HandlerF1, HandlerExactMatch
from src.pipeline.handlers import InterfaceHandler, PredictedAnswer, QuestionAnswerRecipe
from src.utiles import _create_directory_if_not_exist


class Result:

    def __init__(self, question: QuestionAnswerRecipe, predicted_answer: PredictedAnswer):
        self.question = question
        self.predicted_answer = predicted_answer


class HandlerMetricsPerCategory(InterfaceHandler):

    def __init__(self, prefix_dir: str = None, outstream: TextIO = sys.stdout):
        """
        :param prefix_dir: prefix directory for output excels
        :param outstream: a stream for output log
        """
        self.results_by_category: Dict[str, List[Result]] = {}
        self.prefix_dir = prefix_dir if prefix_dir else os.path.join(get_root(), "results", "per_category")
        self.na_statistics_path = f'{self.prefix_dir}/results/na_summary.csv'
        _create_directory_if_not_exist(self.na_statistics_path)
        self.outstream = outstream
        self.metrics_per_category: List[Dict[str, Any]] = []

    def _reset(self):
        self.results_by_category = {}
        self.metrics_per_category = []

    def handle_questions_answers(self, questions: List[QuestionAnswerRecipe], answers: List[PredictedAnswer],
                                 more_info: Dict[str, Any] = {}):
        """
        :param questions: source questions
        :param answers: answers
        :param more_info: "use_tqdm": True|False -- enable / disable tqdm progress bar when processing categories
        :return:
        """
        self._group_by_category(questions, answers, more_info)

        file = open(self.na_statistics_path, 'w')
        for category in sorted(self.results_by_category.keys()):
            self.handle_category(category)
            self.handle_na_category(category, file)
        file.close()

        df = pandas.DataFrame(self.metrics_per_category)
        df.to_excel(os.path.join(self.prefix_dir, "summary.xlsx"), engine="openpyxl")

    def _group_by_category(self, questions: List[QuestionAnswerRecipe], answers: List[PredictedAnswer],
                           more_info: Dict[str, Any]):
        """
        :param questions: questions to be grouped (must be in par with predicted answers)
        :param answers: answers to be grouped (must be in par with questions)
        :param more_info: "category_access_key" -> specify how to get the category info
                          (defaulted to answer.more_info.get(category_access_key")
        :return:
        """
        category_access_key = more_info.get("category_access_key", "predicted_category")

        self._reset()
        for question, answer in zip(questions, answers):
            category = answer.more_info.get(category_access_key, "n/a")
            if category not in self.results_by_category:
                self.results_by_category[category] = []

            res = Result(question, answer)
            self.results_by_category[category].append(res)

    @staticmethod
    def comparator_f1(x: dict) -> float:
        try:
            return float(x["F1"])
        except Exception:
            return -1

    def handle_category(self, category_name: str):

        results = self._get_results(category_name)
        results.sort(key=HandlerMetricsPerCategory.comparator_f1)
        sum_f1 = sum([x["F1"] if x["F1"] else 0.0 for x in results])
        sum_em = sum([x["Exact Match"] if x["Exact Match"] else 0.0 for x in results])
        count = len([x["F1"] for x in results if x["F1"] is not None])
        avg_f1 = sum_f1 / count if count else None
        avg_em = sum_em / count if count else None
        print(f"Cat {category_name} // Count {len(results)} // F1 = {avg_f1} // EM = {avg_em}", file=self.outstream)

        self.metrics_per_category.append(
            {
                "Category": category_name,
                "Count": len(results),
                "F1": avg_f1,
                "Exact match": avg_em
            }
        )

        as_df = pandas.DataFrame(results)
        pathlib.Path(self.prefix_dir).mkdir(parents=True, exist_ok=True)
        as_df.to_excel(os.path.join(self.prefix_dir, f"results_category_{category_name}.xlsx"), engine="openpyxl")
        return results

    def handle_na_category(self, category_name: str, file: TextIO) -> None:
        results = self._get_results(category_name)
        all_na_correct_answer = sum(1 for x in results if x['Actual Answer'] == 'N/A')
        pred_and_correct_na = sum(1 for x in results if x['Actual Answer'] == 'N/A' and x['Predicted Answer'] == None)
        pred_not_na_correct_na = sum(
            1 for x in results if x['Actual Answer'] == 'N/A' and x['Predicted Answer'] != None)
        pred_na_correct_not_na = sum(
            1 for x in results if x['Actual Answer'] != 'N/A' and x['Predicted Answer'] == None)
        print(f"Cat {category_name}, ALL_NA = {all_na_correct_answer}, pred_NA_correct_NA = {pred_and_correct_na}, "
              f"pred_NOT_NA_correct_NA = {pred_not_na_correct_na}, "
              f"pred_NA_correct_NOT_NA = {pred_na_correct_not_na}", file=file)

    def _get_results(self, category_name: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if category_name not in self.results_by_category:
            raise ValueError(f"Bad category name = {category_name}")

        for result in self.results_by_category[category_name]:
            prediction = result.predicted_answer.answer if result.predicted_answer.has_answer() else ""
            truth = result.question.answer if result.question.answer != "N/A" else ""

            f1 = HandlerF1.compute_f1(prediction, truth) if result.question.answer is not None else None
            em = HandlerExactMatch.compute_exact_match(prediction, truth) \
                if result.question.answer is not None else None
            diag = result.predicted_answer.more_info.get("details_for_excel", "")

            answer_row = {
                "PassageId": result.question.recipe.id if result.question.recipe else "N/A",
                "Id": result.question.question_class,
                "Passage": result.question.recipe_passage,
                "Question": result.question.question,
                "Predicted Answer": result.predicted_answer.answer,
                "Actual Answer": result.question.answer,
                "Exact Match": em,
                "F1": f1,
                "Details": diag
            }
            results.append(answer_row)
        return results
