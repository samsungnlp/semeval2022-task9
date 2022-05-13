#!/usr/bin/env python
import pprint

from src.data_question_class_statistics import QuestionClassStatistics
from src.data_statistics import PassageStatsCalculator, QuestionsAnswersMinMaxAvgStats
from src.data_statistics_closed_set_answers import ClosedSetAnswerChecker
from src.data_statistics_extractive_answer import ExtractiveAnswerChecker
from src.data_statistics_extractive_answer import RecoverableAnswerChecker, AppendHandler
from src.datafile_parser import DatafileParser
from src.get_root import get_root

if __name__ == "__main__":

    rec = RecoverableAnswerChecker()
    handler = AppendHandler()
    rec.handler_on_non_recoverable_items.append(handler)
    qcc = QuestionClassStatistics()

    dataset = DatafileParser.get_resource("val")
    checkers = [PassageStatsCalculator(), QuestionsAnswersMinMaxAvgStats(), ExtractiveAnswerChecker(), rec,
                ClosedSetAnswerChecker(), QuestionClassStatistics(), qcc]

    ret = {}
    for checker in checkers:
        metrics = checker.calc_statistics(dataset)
        ret.update(metrics)
    pprint.pprint(ret)

    qcc.save_to_excel_workbook_per_class(f"{get_root()}/resources/")
