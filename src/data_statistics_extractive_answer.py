from sys import stdout
from typing import Any, List, Dict, TextIO

import pandas

from src.data_statistics import DataStatistics, DataItem
from src.data_statistics_closed_set_answers import ClosedSetAnswerChecker
from src.normalizer import Normalizer


class ExtractiveAnswerChecker(DataStatistics):

    def __init__(self):
        self.normalizer = Normalizer()
        self.num_extractive_and_nonextractive_answers = 0  # not N/As and not integers
        self.num_extractive_answers = 0
        self.num_non_extractive_answers = 0
        self.turns_extractive_after_removing_article = 0
        self.turns_extractive_after_removing_heading_conjunction = 0

    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:

        for item in data:
            self.process_item(item)

        ret = {
            "num_extractive_and_nonextractive_answers": self.num_extractive_and_nonextractive_answers,
            "num_extractive_answers": self.num_extractive_answers,
            "num_non_extractive_answers": self.num_non_extractive_answers,
        }

        return ret

    def process_item(self, item: DataItem):

        if not ExtractiveAnswerChecker.is_textual_answer(item.answer):
            return
        self.num_extractive_and_nonextractive_answers += 1

        passage = self.normalizer.normalize(item.passage)
        answer = self.normalizer.normalize(item.answer)

        if passage.find(answer) != -1:
            self.num_extractive_answers += 1
        else:
            self.num_non_extractive_answers += 1

    @staticmethod
    def is_textual_answer(text: str) -> bool:
        return text is not None \
               and not text.isnumeric() \
               and text not in {"N/A", "the first event", "the second event"}


class AppendHandler:
    def __init__(self):
        self.append_items: List[DataItem] = []

    def handle(self, item: DataItem) -> None:
        self.append_items.append(item)

    def dump_to_out(self, outstream: TextIO = stdout):
        for item in self.append_items:
            print(f"{item.question}\t\t{item.answer}", file=outstream)

    def dump_to_excel(self, filename: str):
        df = pandas.DataFrame([item.to_dataframe_dict() for item in self.append_items], columns=DataItem.COLUMNS)
        df.to_excel(filename)


class RecoverableAnswerChecker(DataStatistics):

    def __init__(self):
        self.normalizer = Normalizer()
        self.recoverable__analysed_answers = 0
        self.recoverable__after_removing_article = 0
        self.recoverable__after_removing_heading_by = 0
        self.recoverable__non_recoverable = 0
        self.recoverable__skipped_answers = 0

        self.recoverable_dict: Dict[str, int] = {}

        self.handler_on_analysed_items = []
        self.handler_on_non_recoverable_items = []
        self.prefixes = ["by using", "by adding", "by mixing", "by", "when", "until", "with"]

        for prefix in self.prefixes:
            self.__reset_counter(prefix)

    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:

        for item in data:
            self.process_item(item)

        ret = {
            "recoverable__skipped_answers": self.recoverable__skipped_answers,
            "recoverable__analysed_answers": self.recoverable__analysed_answers,
            "recoverable__after_removing_article": self.recoverable__after_removing_article,
            "recoverable__non_recoverable": self.recoverable__non_recoverable,
        }
        ret.update(self.recoverable_dict)

        return ret

    def process_item(self, item: DataItem):

        if not ExtractiveAnswerChecker.is_textual_answer(item.answer):
            self.recoverable__skipped_answers += 1
            return

        passage = self.normalizer.normalize(item.passage)
        answer = self.normalizer.normalize(item.answer)

        if ClosedSetAnswerChecker.is_closed_aswer(answer) or passage.find(answer) != -1:
            self.recoverable__skipped_answers += 1
            return

        self.recoverable__analysed_answers += 1

        self._handle_analysed_item(item)

        answer, passage = self._remove_articles(answer, passage)

        if passage.find(answer) != -1:
            self.recoverable__after_removing_article += 1
            return

        for prefix in self.prefixes:
            ans_without_prefix = answer.replace(prefix, "", 1).strip()
            if passage.find(ans_without_prefix) != -1:
                self.__increment_counter(prefix)
                return

        self._handle_non_recoverable_item(item)
        self.recoverable__non_recoverable += 1

    def _handle_analysed_item(self, item):
        for handler in self.handler_on_analysed_items:
            handler.handle(item)

    def _handle_non_recoverable_item(self, item):
        for handler in self.handler_on_non_recoverable_items:
            handler.handle(item)

    def _remove_articles(self, answer, passage):
        for article in ["a", "an", "the"]:
            answer = answer.replace(article, " ")
            passage = passage.replace(article, " ")
        answer = self.normalizer.normalize(answer)
        passage = self.normalizer.normalize(passage)
        return answer, passage

    def __increment_counter(self, phrase: str):
        phrase = phrase.replace(" ", "_")
        key = f"recoverable__after_removing_heading_{phrase}"
        if key in self.recoverable_dict:
            self.recoverable_dict[key] += 1
        else:
            self.recoverable_dict[key] = 1

    def __reset_counter(self, phrase: str):
        phrase = phrase.replace(" ", "_")
        key = f"recoverable__after_removing_heading_{phrase}"
        self.recoverable_dict[key] = 0
