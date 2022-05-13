from typing import Any, Dict, List, TextIO
import sys

from src.pipeline.handlers import InterfaceHandler, QuestionAnswerRecipe, PredictedAnswer


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class HandlerF1(InterfaceHandler):

    def __init__(self, output_stream: TextIO = sys.stdout):
        self.output_stream = output_stream
        self.last_stored_f1 = None

    def handle_questions_answers(self, questions: List[QuestionAnswerRecipe], answers: List[PredictedAnswer],
                                 more_info: Dict[str, Any] = {}):
        """
        :param questions: source questions with expected answers (supports missing answers)
        :param answers: predicted answers
        :param more_info: skipped
        :return: N/A, prints final results to the output
        """

        sum_f1 = 0.0
        count_f1 = 0
        for question, predicted_answer in zip(questions, answers):

            if not question.answer:
                continue

            prediction = predicted_answer.answer if predicted_answer.has_answer() else ""
            truth = question.answer if question.answer != "N/A" else ""
            sum_f1 += HandlerF1.compute_f1(prediction, truth)
            count_f1 += 1

        self.last_stored_f1 = sum_f1 / count_f1 if count_f1 else "N/A"
        print(f"F1 = {self.last_stored_f1}", file=self.output_stream)

    @staticmethod
    def compute_f1(prediction: str, truth: str) -> float:
        """
        copied from:
        https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html
        """
        pred_tokens = normalize_text(prediction).split()
        truth_tokens = normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = []
        extra_pred_tokens = pred_tokens.copy()
        for token in truth_tokens:
            if token in extra_pred_tokens:
                common_tokens.append(token)
                extra_pred_tokens.remove(token)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)


class HandlerExactMatch(InterfaceHandler):

    def __init__(self, output_stream: TextIO = sys.stdout):
        self.output_stream = output_stream
        self.last_result = None

    def handle_questions_answers(self, questions: List[QuestionAnswerRecipe], answers: List[PredictedAnswer],
                                 more_info: Dict[str, Any] = {}):
        """
        :param questions: source questions with expected answers (supports missing answers)
        :param answers: predicted answers
        :param more_info: skipped
        :return: N/A, prints final results to the output
        """
        exact_match_sum = 0.0
        exact_match_count = 0
        for question, predicted_answer in zip(questions, answers):
            if not question.answer:
                continue

            prediction = predicted_answer.answer if predicted_answer.has_answer() else ""
            truth = question.answer if question.answer != "N/A" else ""
            exact_match_sum += HandlerExactMatch.compute_exact_match(prediction, truth)
            exact_match_count += 1

        self.last_result = exact_match_sum / exact_match_count if exact_match_count else "N/A"
        print(f"Exact match = {self.last_result}", file=self.output_stream)

    @staticmethod
    def compute_exact_match(prediction: str, truth: str) -> float:
        """
        implementation  from:
        https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html
        """
        return int(normalize_text(prediction) == normalize_text(truth))
