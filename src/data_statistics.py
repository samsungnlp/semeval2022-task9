import abc
from typing import List, Dict, Any, Set

from src.datafile_parser import DataItem


class DataStatistics:
    INF = 1_000_000_000

    @abc.abstractmethod
    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:  # pragma: nocover
        pass


class QuestionsAnswersMinMaxAvgStats(DataStatistics):

    def __init__(self):
        self.num_questions = 0

        self.num_answers = 0
        self.num_empty_answers = 0
        self.num_textual_answers = 0
        self.num_na_answers = 0
        self.num_integer_answers = 0
        self.num_first_second_event_answers = 0

        self.min_question_len = 0
        self.max_question_len = 0
        self.sum_question_len = 0

        self.min_answer_len = 0
        self.max_answer_len = 0
        self.sum_answer_len = 0
        self.reset()

    def reset(self):
        self.num_questions = 0

        self.num_answers = 0
        self.num_empty_answers = 0
        self.num_textual_answers = 0
        self.num_na_answers = 0
        self.num_integer_answers = 0
        self.num_first_second_event_answers = 0

        self.min_question_len = DataStatistics.INF
        self.max_question_len = 0
        self.sum_question_len = 0

        self.min_answer_len = DataStatistics.INF
        self.max_answer_len = 0
        self.sum_answer_len = 0

    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:
        self.reset()

        for item in data:
            self.process_item(item)

        return {
            "num_questions": self.num_questions,
            "num_answers": self.num_answers,
            "num_empty_answers": self.num_empty_answers,
            "num_na_answers": self.num_na_answers,
            "num_int_answers": self.num_integer_answers,
            "num_textual_answers": self.num_textual_answers,
            "num_first_second_event_answers": self.num_first_second_event_answers,
            "min_answer_len": self.min_answer_len,
            "max_answer_len": self.max_answer_len,
            "min_question_len": self.min_question_len,
            "max_question_len": self.max_question_len,
            "avg_question_len": self.sum_question_len / self.num_questions if self.num_questions else None,
            "avg_answer_len": self.sum_answer_len / self.num_answers if self.num_answers else None,
        }

    def process_item(self, item: DataItem):
        self.num_questions += 1
        self.sum_question_len += len(item.question)
        self.min_question_len = min(len(item.question), self.min_question_len)
        self.max_question_len = max(len(item.question), self.max_question_len)

        if item.answer:
            self.num_answers += 1

        if not item.has_answer():
            self.num_empty_answers += 1
        elif item.answer == "N/A":
            self.num_na_answers += 1
        elif item.answer in {"the first event", "the second event"}:
            self.num_first_second_event_answers += 1
        elif item.answer.isnumeric():
            self.num_integer_answers += 1
        else:
            self.num_textual_answers += 1

        if item.has_answer():
            ans_len = len(item.answer)
            self.sum_answer_len += ans_len
            self.min_answer_len = min(ans_len, self.min_answer_len)
            self.max_answer_len = max(ans_len, self.max_answer_len)


class PassageStatsCalculator(DataStatistics):

    def __init__(self):
        self.num_passages = 0
        self.min_passage_chars = DataStatistics.INF
        self.max_passage_chars = 0
        self.sum_passage_chars = 0

        self.max_passage_lines = 0
        self.min_passage_lines = DataStatistics.INF
        self.sum_passage_lines = 0
        self.unique_passages: Set[str] = set()

    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:
        for item in data:
            self.process_item(item)

        return {
            "num_passages": self.num_passages,
            "num_unique_passages": len(self.unique_passages),

            "min_passage_chars": self.min_passage_chars,
            "max_passage_chars": self.max_passage_chars,
            "sum_passage_chars": self.sum_passage_chars,
            "avg_passage_chars": self.sum_passage_chars / self.num_passages if self.num_passages else None,

            "max_passage_lines": self.max_passage_lines,
            "min_passage_lines": self.min_passage_lines,
            "sum_passage_lines": self.sum_passage_lines,
            "avg_passage_lines": self.sum_passage_lines / self.num_passages if self.num_passages else None,
        }

    def process_item(self, item: DataItem):
        if item.passage is None:
            return

        self.num_passages += 1
        self.unique_passages.add(item.passage)
        passage_len = len(item.passage)
        self.min_passage_chars = min(self.min_passage_chars, passage_len)
        self.max_passage_chars = max(self.max_passage_chars, passage_len)
        self.sum_passage_chars += passage_len

        passage_lines = len([line for line in item.passage.split("\n") if line.strip()])
        self.min_passage_lines = min(self.min_passage_lines, passage_lines)
        self.max_passage_lines = max(self.max_passage_lines, passage_lines)
        self.sum_passage_lines += passage_lines
