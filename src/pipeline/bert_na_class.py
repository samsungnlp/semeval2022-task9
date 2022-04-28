import json
import os

from src.get_root import get_root


class NAQuestionClassifier():
    def __init__(self, predictions_path: str = "data/bert_na_score_test.json"):
        with open(os.path.join(get_root(), predictions_path), "r") as f:
            self.na_predictions = json.load(f)

    def print_it(self):
        print(self.na_predictions)

if __name__ == "__main__":
    check = NAQuestionClassifier()
    check.print_it()
