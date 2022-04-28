from typing import List, Dict, Any

from src.data_statistics import DataStatistics, DataItem
from src.normalizer import Normalizer


class ClosedSetAnswerChecker(DataStatistics):
    DEFAULT_ANSWERS = ["n/a", "N/A", "n a", "the first event", "the second event", "bowl", "pan", "by using a knife",
                       "by using a grater", "by using a lid", "by using a spoon", "by using a spatula",
                       "by using a mixer", "by using a rolling pin", "by using a peeler", "by hand"]

    def __init__(self, closed_answers: List[str] = DEFAULT_ANSWERS):
        self.normalizer = Normalizer()
        self.closed_answers = closed_answers
        self.found_answers: Dict[str, int] = {}

        for item in self.closed_answers:
            self.__reset_counter(item)
        self.found_answers["closed_answers_remaining_answers"] = 0

    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:
        for item in data:
            self.process_item(item)

        return self.found_answers

    def process_item(self, item: DataItem):
        answer = item.answer

        for item in self.closed_answers:
            closed_answer = self.normalizer.normalize(item)
            if closed_answer == answer:
                self.__increment_counter(item)
                return

        self.found_answers["closed_answers_remaining_answers"] += 1

    def __increment_counter(self, phrase: str):
        phrase = phrase.replace(" ", "_")
        key = f"closed_answers_{phrase}"
        if key in self.found_answers:
            self.found_answers[key] += 1
        else:
            self.found_answers[key] = 1

    def __reset_counter(self, phrase: str):
        phrase = phrase.replace(" ", "_")
        key = f"closed_answers_{phrase}"
        self.found_answers[key] = 0

    @staticmethod
    def is_closed_aswer(answer: str) -> bool:
        return answer.lower().strip() in ClosedSetAnswerChecker.DEFAULT_ANSWERS
