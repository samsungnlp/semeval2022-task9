import io
from typing import Any, Dict, List

from src.annotated_recipe import AnnotatedSentence, AnnotatedToken
from src.pipeline.interface_question_answering import QuestionAnsweringBase, PredictedAnswer, QuestionAnswerRecipe
from src.pipeline.question_category import QuestionCategory


class QuestionAnswerer14Preheat(QuestionAnsweringBase):

    def __init__(self):
        self.sink = io.StringIO()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        extra_info = {"source": "QuestionAnswerer: How do you preheat?"}

        question_as_array = question.question.replace("?", "").split(" ")
        verb = question_as_array[3].lower()
        obj = [x.lower() for x in question_as_array[4:] if x not in {"the", "a", "an", "your"}]
        obj = obj[-1]
        print(f"V = {verb}\nO = {obj}", file=self.sink)

        candidates = []
        for sentence in question.recipe.annotated_recipe.annotated_sentences:
            normalized_tokens = [t.raw_token.lower() for t in sentence.annotated_tokens]

            if not (verb in normalized_tokens and obj in normalized_tokens):
                continue

            print(f"Sentence = {sentence.raw_sentence}", file=self.sink)
            goal_tokens = self.get_goals_from_sentence(sentence)
            print([g.raw_token for g in goal_tokens], file=self.sink)
            candidate = " ".join([g.raw_token for g in goal_tokens])
            candidates.append(candidate)

        print(f"Candidates = {candidates}", file=self.sink)
        ret = " ".join(question_as_array[3:] + [candidates[0]]) if len(candidates) == 1 else None

        print(f"Final answer = {ret}", file=self.sink)
        print(f"Truth = {question.answer}", file=self.sink)

        # if question.answer and ret and question.answer.lower() != ret.lower():
        #     print(self.sink.getvalue())
        #     self.sink = io.StringIO()

        return PredictedAnswer(ret, raw_question=question.question, more_info=extra_info)

    def get_goals_from_sentence(self, sentence: AnnotatedSentence) -> List[AnnotatedToken]:
        return [t for t in sentence.annotated_tokens if "B-Goal" in t.semantic_roles or "I-Goal" in t.semantic_roles]
