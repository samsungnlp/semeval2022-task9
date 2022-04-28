from typing import Dict, Any, Tuple, List

from src.annotated_recipe import AnnotatedRecipe, AnnotatedToken
from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
import inflect

from src.putty_lemmatizer import PuttyLemmatizer


class QuestionAnswerer3What(QuestionAnsweringBase):
    # TODO: Apply some of these ideas for improvement:

    DESCRIPTION = "QuestionAnswerer: 03_what_is_in?"

    def __init__(self):
        self.inflect_engine = inflect.engine()
        self.inflect_engine.classical()
        self.lemmatizer = PuttyLemmatizer()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        Answers the question "What's in the ...", belonging to class 3

        :param question: question to be answered
        :param question_category: assumed to be 3
        :param more_info: ignored
        :return: answer about contents
        """
        more_info_for_answer = {"source": QuestionAnswerer3What.DESCRIPTION}

        question_spans = self.extract_essential_span_from_question(question.question)

        starts = []
        for span in question_spans:
            starts.extend(self.find_answer_start(span, question.recipe.annotated_recipe))

        if len(starts) > 1 and len(question_spans) > 1:
            starts = starts[:1]

        potential_answers = []
        for start in starts:
            answer = self.retrieve_answer(question.recipe.annotated_recipe, start[0], start[1])
            if answer:
                potential_answers.append(answer)
        answer = potential_answers[-1] if potential_answers else None
        return PredictedAnswer(answer, raw_question=question.question, confidence=None,
                               more_info=more_info_for_answer)

    @staticmethod
    def extract_essential_span_from_question(question: str) -> List[str]:
        """
        Extracts the part after "What's in the"
        """
        question_beginning = "What's in the"
        span = question[len(question_beginning) + 1:-1]
        splitted_by_and = span.split(" and ")
        spans = [span]
        if splitted_by_and[0] != span:
            spans.extend(splitted_by_and)
        return list(dict.fromkeys(spans))

    def find_answer_start(self, question_span: str, recipe: AnnotatedRecipe) -> List[Tuple[int, int]]:
        """
        Finds in recipe a token which starts the answer
        """
        span = question_span.replace(" ", "_")

        matching_sentence_and_token_pairs = []
        for sentence in recipe.annotated_sentences:
            if "ingredients" in sentence.sentence_id:
                continue

            for token in sentence.annotated_tokens:
                if self.token_match(token, span):
                    matching_sentence_and_token_pairs.append((sentence.sentence_position_in_paragraph, token.id - 1))
                    break

        return matching_sentence_and_token_pairs

    @staticmethod
    def retrieve_answer(recipe: AnnotatedRecipe, sentence_idx: Any, token_idx: Any) -> Any:
        """
        Composes an answer from the sentence's contents
        """
        if sentence_idx is None:
            return None

        allowed_roles = {"I-Patient", "B-Patient", "B-Co-Patient", "I-Co-Patient", "I-Theme", "B-Theme", "I-Co-Theme", None}

        sentence = recipe.annotated_sentences[sentence_idx]

        relevant_tokens = sentence.annotated_tokens[token_idx:]
        verb_position = QuestionAnswerer3What.which_in_turn_verb_in_sentence(relevant_tokens[0])

        contents = []

        ingredient_continuation = False
        for token in relevant_tokens[1:]:
            if "INGREDIENT" not in token.role_in_recipe:
                ingredient_continuation = False
                continue

            if not ingredient_continuation and not token.is_equal_to_any_verb_id(relevant_tokens[0].id):
                continue

            role = None if verb_position is None else token.semantic_roles[verb_position - 1]  # because they are 0-based
            if role in allowed_roles:
                if ingredient_continuation:
                    contents[-1] += " " + token.raw_token
                else:
                    contents.append(token.raw_token)
                    ingredient_continuation = True

        drop_information = relevant_tokens[0].get_entry_from_relation1("Drop")
        if drop_information:
            drop_phrases = [elem.replace("_", " ") for elem in drop_information]
            contents = QuestionAnswerer3What.remove_duplicates_from_drop(contents, drop_phrases)
            contents += drop_phrases

        if not contents:
            return None

        answer = QuestionAnswerer3What.concatenate_words_into_phrase(contents)

        return answer if answer.startswith("the ") else "the " + answer

    @staticmethod
    def concatenate_words_into_phrase(words: List[str]) -> str:
        """
        Transforms a list of the from ["x", "y", "z"] to the string "x, y and z"
        :param words: The words to concatenate
        :return: The phrase
        """

        if not words:
            return ""
        if len(words) == 1:
            return words[0]

        return ", ".join(words[:-1]) + " and " + words[-1]

    def token_match(self, token: AnnotatedToken, span: str) -> bool:
        if token.relation1 is None:
            return False

        for entry in token.get_entry_from_relation1("Result"):
            entry_normalised = entry.lower().strip("_")
            if span == entry_normalised:
                return True
            entry_singular = self.inflect_engine.singular_noun(entry_normalised.replace("_", " "))
            entry_singular = entry_singular.replace(" ", "_") if entry_singular else ""

            if any(word in [entry_singular, entry_normalised] for word in [span, self.lemmatize_span(span)]):
                return True
            return False

    @staticmethod
    def which_in_turn_verb_in_sentence(verb: AnnotatedToken) -> int:
        """
        For example, for sentence "Walk, run for a while, skip and take a break." and verb "skip" returns 3
        :param verb: The annotated verb
        :return: The position in sentence or 1 if not found
        """

        for i, role in enumerate(verb.semantic_roles):
            if role == "B-V":
                return i + 1

        return 1

    @staticmethod
    def remove_duplicates_from_drop(ingredients: List[str], drop: List[str]) -> List[str]:
        unique = []

        for ingredient in ingredients:
            if any(drop_phrase.startswith(ingredient) for drop_phrase in drop):
                continue
            unique.append(ingredient)

        return unique

    def lemmatize_span(self, span: str) -> str:
        return "_".join(self.lemmatizer.lemmatize_noun(word) for word in span.split("_"))
