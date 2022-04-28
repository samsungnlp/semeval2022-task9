from typing import Union

from src.get_root import get_root
from src.unpack_data import QuestionAnswerRecipe
from src.utiles import read_dict


class BertAnswerNA:
    def __init__(self, which: str):
        self.which = which
        path_bert_na = f'{get_root()}/data/bert_na_score_{self.which}.json'
        self.bert_na_data = read_dict(path_bert_na)

    def check_bert_na_answer(self, answer: Union[str, None], question: QuestionAnswerRecipe) -> Union[str, None]:
        if self.which != 'train' and answer:
            if self.bert_na_data.get(f'{question.recipe.id}-{question.question_class}', 0) > 0.99:
                print(f'FIX BY BERT: ANSWER BEFORE BERT: {answer}, CORRECT ANSWER: {question.answer}')
                return None
        return answer
