from src.pipeline.interface_question_answering import QuestionAnswerRecipe
from src.annotated_recipe import AnnotatedSentence
from src.unpack_data import Recipe
from typing import Dict, List, Tuple
import inflect
from src.utils_class_method import WordMistakesRepair


class QuestionAnswererMethodAttribute:
    DESCRIPTION = "QuestionAnswerer: How do you?"

    def __init__(self, semantic_roles: List[str], answer_annotations: List[str]):
        self.semantic_roles = semantic_roles
        self.answer_annotations = answer_annotations

        self.inflection = inflect.engine()
        self.mistakes = WordMistakesRepair()

    @staticmethod
    def concat_words(sentence: AnnotatedSentence, annotation_column: int,
                     possible_answer: str, raw: bool = False) -> Dict[tuple, str]:
        """
        Concatenate words with the same semantic annotation (B-XX with I-XX)
        :param sentence: annotated sentence being analyzed
        :param annotation_column: column index where we look for semantic roles
        :param possible_answer: semantic role name we are looking for (like "V" in "B-V")
        :param raw: True if we want to return raw (not normalized) token
        :return: a dictionary indexed by (sentence id, paragraph id, token index) with corresponding words
        """

        words = {}
        extra_idx = None
        b_answer_found = False

        for token_idx, token in enumerate(sentence.annotated_tokens):
            value = token.raw_token.lower() if raw else token.normalized_token.lower()
            semantic_role = token.semantic_roles[annotation_column]

            if semantic_role is None:
                continue

            # First we look for f"B-{possible_answer}
            if semantic_role == f"B-{possible_answer}":
                b_answer_found = True
                extra_idx = token.id
                words[(sentence.sentence_id, sentence.paragraph_id, extra_idx)] = [value.lower()]

            # If f"B-{possible_answer}" has been found, we append f"I-{possible_answer}"s if we find them
            elif b_answer_found and semantic_role == f"I-{possible_answer}":
                words[(sentence.sentence_id, sentence.paragraph_id, extra_idx)].append(value.lower())

        words = {key: " ".join(value) for key, value in words.items()}
        return words

    def words_from_paragraph(self, recipe: Recipe, paragraph: str, used_column: int,
                             searched_word: str, raw: bool) -> Dict[tuple, str]:
        """
        Collects all words with the same semantic annotation within the paragraph
        """

        words_in_paragraph = {}
        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            if sentence.sentence_id == paragraph:
                words_in_paragraph.update(self.concat_words(sentence, used_column, searched_word, raw))

        return words_in_paragraph

    def cut_rows_and_answer(self, verb: str, v_object: str, recipe: Recipe, steps: List[str], column: int,
                            answer_annotations: List[str] = None) -> str:
        """
        Iterate over possible answer annotation and search for an answer
        """
        if not answer_annotations:
            answer_annotations = self.answer_annotations

        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            if sentence.sentence_id in steps:
                for annotation in answer_annotations:
                    answer_dict = self.concat_words(sentence, column, annotation, True)
                    if answer_dict:
                        answer = list(answer_dict.values())[0]
                        return f"{verb} {v_object} {answer}"
        return ""

    def semantic_iteration(self, verb: str, semantic_role_examples: Dict[tuple, str], used_column: int,
                           recipe: Recipe, question: str, verb_steps: List[str]) -> str:
        """
        Creates intersection of steps in which there is a verb and semantic example
        """
        for example in semantic_role_examples.values():
            if example.lower() in question:
                example_steps = [idx[0] for idx, example_value in semantic_role_examples.items() if
                                 example_value == example]

                intersection = [value for value in verb_steps if value in example_steps]

                if intersection and intersection[0]:
                    answer = self.cut_rows_and_answer(verb, example, recipe, intersection, used_column)

                    if answer:
                        return answer
        return ""

    @staticmethod
    def search_drop_column(recipe: Recipe, verb: str, verb_steps: List[str]) -> List[str]:
        """
        Get value from drop column for verb
        """
        relations = []
        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            if sentence.sentence_id in verb_steps:
                for token in recipe.annotated_recipe.annotated_sentences[sentence_idx].annotated_tokens:
                    if token.normalized_token.lower() in verb and token.relation1:
                        relations.append(token.relation1)
        return relations

    def make_use_of_drop(self, drop_column: str) -> str:
        drop_column = drop_column.split("|")[0].split("=")[1:][0].split(":")
        drop_column = [value.split(".")[0] for value in drop_column]
        drop_column = [" ".join(value.split("_")) for value in drop_column]
        drop_column = [value.replace(" - ", "-") for value in drop_column]
        drop_column = [self.inflection.singular_noun(value) if self.inflection.singular_noun(value)
                       else value for value in drop_column]
        drop_column = [value.replace("leaf", "leave") for value in drop_column]
        drop_column = ", ".join(drop_column)
        drop_column = drop_column.rsplit(",", 1)
        drop_column = " and".join(drop_column)
        drop_column = f"the {drop_column}"
        return drop_column.lower()

    def verb_iteration(self, used_column: int, recipe: Recipe, question: str, verbs: Dict[tuple, str],
                       semantic_role_examples: Dict[tuple, str] = {}, use_drop_column: bool = False,
                       semantics: bool = True) -> str:
        """
        Iterating over all of the verbs in step
        """
        for verb in verbs.values():
            if verb in question:
                verb_steps = [idx[0] for idx, verb_value in verbs.items() if verb_value == verb]
                if use_drop_column:
                    drop_column_values = self.search_drop_column(recipe, verb, verb_steps)
                    for drop_column_value in drop_column_values:
                        if drop_column_value and ("Drop" in drop_column_value):
                            drop_value = self.make_use_of_drop(drop_column_value)
                            if drop_value in question:
                                answer = self.cut_rows_and_answer(verb, drop_value, recipe, verb_steps, used_column)
                                if answer:
                                    return answer
                            if drop_value.replace("-", " - ") in question:
                                answer = self.cut_rows_and_answer(
                                    verb, drop_value.replace("-", " - "), recipe, verb_steps, used_column)
                                if answer:
                                    return answer

                else:
                    if semantics:
                        answer = self.semantic_iteration(verb, semantic_role_examples, used_column, recipe,
                                                         question, verb_steps)
                        if answer:
                            return answer
                    else:
                        answer = self.cut_rows_and_answer(verb, "", recipe, verb_steps, used_column)
                        if answer:
                            return answer

        return ""

    def steps_and_columns_iteration(self, recipe: Recipe, question: str, paragraphs: List[str], semantics: bool = True,
                                    semantic_roles: List[str] = None, use_drop_column: bool = False) -> str:
        """
        Iterate over paragraphs and columns
        """
        if not semantic_roles:
            semantic_roles = self.semantic_roles

        for paragraph in paragraphs:
            for used_column in range(0, 10):

                verbs = self.words_from_paragraph(recipe, paragraph, used_column, "V", False)
                if use_drop_column:
                    answer = self.verb_iteration(used_column, recipe, question, verbs, use_drop_column=True)
                    if answer:
                        return answer

                else:
                    if semantics:
                        for semantic_role in semantic_roles:
                            semantic_role_examples = self.words_from_paragraph(recipe, paragraph, used_column,
                                                                               semantic_role, True)
                            answer = self.verb_iteration(used_column, recipe, question, verbs, semantic_role_examples)
                            if answer:
                                return answer
                    else:
                        answer = self.verb_iteration(used_column, recipe, question, verbs, semantics=False)
                        if answer:
                            return answer

        return ""

    def answer_method_attribute_question(self, question: QuestionAnswerRecipe) -> Tuple[str, Dict[str, str]]:

        """
        Answers the question "how do you", belonging to class method / attribute (original 10).
        Iterate over paragraphs and semantic role's columns to find specified labels and relations.
        :param question: question to be answered
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswererMethodAttribute.DESCRIPTION}
        paragraphs = [sentence.sentence_id for sentence in question.recipe.annotated_recipe.annotated_sentences]
        question_changed = self.mistakes.change_question(question.question.lower())

        answer = self.steps_and_columns_iteration(question.recipe, question_changed, paragraphs,
                                                  semantics=False, use_drop_column=True)
        if answer:
            more_info_for_answer["details_for_excel"] = "relation || class Method/Attribute"
            return self.mistakes.change_answer(question.question.lower(), answer), more_info_for_answer

        answer = self.steps_and_columns_iteration(question.recipe, question_changed, paragraphs)

        if answer:
            more_info_for_answer["details_for_excel"] = "semantics || class Method/Attribute"
            return self.mistakes.change_answer(question.question.lower(), answer), more_info_for_answer

        more_info_for_answer["details_for_excel"] = "no match || class Method/Attribute"
        return "", more_info_for_answer
