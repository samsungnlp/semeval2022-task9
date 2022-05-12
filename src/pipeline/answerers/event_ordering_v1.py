import re
from typing import Dict, Any, Union, Tuple

import jellyfish

from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory


class QuestionAnswererEventOrderingV1(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: ...which comes first?"

    def __init__(self):
        self.question: QuestionAnswerRecipe = None

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: assumed to be event ordering (not checked!)
        :param more_info: ignored
        :return: the answer
        """
        self.question = question
        more_info_for_answer = {"source": QuestionAnswererEventOrderingV1.DESCRIPTION}

        return PredictedAnswer(_return_answer(question, 0.8), raw_question=question.question, more_info=more_info_for_answer)


def _return_answer(question, threshold: float) -> Union[str, None]:
    first_compare = []
    second_compare = []
    q_first, q_second = _separate_first_and_second_question_part(question.question)
    for k, v in question.recipe.new_pars.items():
        if 'step' in k:
            first_compare.append(_distance_scores(q_first, v[0].metadata['text']))
            second_compare.append(_distance_scores(q_second, v[0].metadata['text']))
    f_nr = first_compare.index(min(first_compare))
    s_nr = second_compare.index(min(second_compare))
    answer = None
    if min(first_compare) > threshold - 0.04 or min(second_compare) > threshold:
        return None
    if f_nr > s_nr:
        answer = 'the second event'
    if f_nr < s_nr:
        answer = 'the first event'

    return answer


def _separate_first_and_second_question_part(question: str) -> Tuple[str, str]:
    and_start_indexes = [r.start() for r in re.finditer(' and ', question)]
    before_and = ''
    for and_start_index in and_start_indexes:
        if question[and_start_index - 1] != ',':
            before_and = question[and_start_index - 1]
            break
    splinted_q = question.split(f'{before_and} and ')
    return f'{splinted_q[0]}{before_and}', splinted_q[1].split(', which comes first?')[0]


def _distance_scores(item_first: str, item_second: str) -> float:
    """
    score as maximum distance (1.0), best(0.0)
    """
    edit_distance_fn = lambda x, y: 1 - jellyfish.jaro_similarity(x, y)
    return edit_distance_fn(_clean(item_first), _clean(item_second))


def _clean(str_to_clean: str) -> str:
    str_to_clean = str_to_clean.lower()
    for token in [r'(?:\b(?:a|an|the|le)\b)', r'^\b(a|an|the|le)\b']:
        str_to_clean = re.sub(token, '', str_to_clean)
    return str_to_clean.replace(' - ', ' ')
