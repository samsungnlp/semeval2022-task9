from src.annotated_recipe import AnnotatedSentence
from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.unpack_data import Recipe
from src.pipeline.question_category import QuestionCategory
from typing import Dict, List, Any, Optional
import inflect


class QuestionAnswererUniversalSrl(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: Universal for SRL answers: Extent, Result, Time, Purpose, Co-patient, " \
                  "Location SRL, Source"

    def __init__(self, semantic_roles: List[str], answer_annotations: List[str], reversed_paragraphs: bool = True):
        self.semantic_roles = semantic_roles
        self.answer_annotations = answer_annotations
        self.reversed_paragraphs = reversed_paragraphs

        self.inflection = inflect.engine()

    @staticmethod
    def concat_words(sentence: AnnotatedSentence, annotation_column: int,
                     possible_answer: str, raw: bool = False) -> Dict[tuple, str]:
        """
        Create a dictionary indexed by (sentence_id, paragraph_id, token_index)
        containing words having the f"B-{possible_answer}" and f"I-{possible_answer} semantic role
        A recipe sentence consists of a list of tokens;
            iterate them looking for f"B-{possible_answer}" semantic role
        Tokens are augmented with a list of semantic roles; only look at roles indexed with annotation_column
        (Imagine the semantic roles as a 2D array indexed by token index and semantic role index)
        If f"B-{possible_answer}" is found, add a dictionary entry
        When subsequent f"I-{possible_answer}s are found, append the values to the dictionary entry,
            using the f"B-{possible_answer}" token index found previously
        Remarks:
            If raw take the raw token value, else normalized; then take lowercase
            Words having the same dictionary index are further concatenated
        :param sentence: annotated recipe sentence being analyzed
        :param annotation_column: token semantic role index to look at
        :param possible_answer: semantic role name we are looking for (like "V" in "B-V", "I-V")
        :param raw: True if we want to return raw (not normalized) token value
        :return: a dictionary indexed by (sentence id, paragraph id, token index) with corresponding words
        """

        words_dict = {}
        b_role_idx = None  # Index of the token having the f"B-{possible_answer}" semantic role
        sentence_id = sentence.sentence_id
        paragraph_id = sentence.paragraph_id
        b_answer_found = False

        for token_idx, token in enumerate(sentence.annotated_tokens):

            # If raw take the raw token value, else normalized; then take lowercase
            value = token.raw_token.lower() if raw else token.normalized_token.lower()
            # Only look at semantic roles indexed with annotation_column
            semantic_role = token.semantic_roles[annotation_column]

            # First we look for f"B-{possible_answer}"
            if semantic_role == f"B-{possible_answer}":
                b_answer_found = True
                b_role_idx = token_idx
                words_dict[(sentence_id, paragraph_id, b_role_idx)] = [value]

            # When subsequent f"I-{possible_answer}s are found, append the values to the dictionary entry,
            # using the f"B-{possible_answer}" token index found previously
            elif b_answer_found and semantic_role == f"I-{possible_answer}":
                words_dict[(sentence_id, paragraph_id, b_role_idx)].append(value)

        # Words having the same dictionary index are further concatenated and presented as string
        words_dict = {key: " ".join(value) for key, value in words_dict.items()}

        return words_dict

    def words_from_paragraph(self, recipe: Recipe, paragraph: str, annotation_column: int,
                             searched_role: str, raw: bool) -> Dict[tuple, str]:
        """
        Collects all words with the required semantic annotation within the paragraph
        See the documentation for concat_words for explanations for annotation_column and searched_role
        """

        words_in_paragraph = {}
        for sentence in recipe.annotated_recipe.annotated_sentences:
            if sentence.paragraph_id == paragraph:
                words_in_paragraph.update(self.concat_words(sentence, annotation_column, searched_role, raw))

        return words_in_paragraph

    def cut_rows_and_answer(self, recipe: Recipe, steps: List[str], iter_type: str, column: int,
                            answer_annotations: Optional[List[str]] = None) -> str:
        """
        Iterate over possible answer annotation and search for an answer
        """
        if not answer_annotations:
            answer_annotations = self.answer_annotations

        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):

            part_identifier = sentence.sentence_id if iter_type == "sentence" else sentence.paragraph_id

            if part_identifier in steps:
                for annotation in answer_annotations:
                    answer_dict = self.concat_words(sentence, column, annotation, True)
                    if answer_dict:
                        answer = list(answer_dict.values())[0]
                        return answer
        return ""

    def semantic_iteration(self, semantic_role_examples: Dict[tuple, str], annotation_column: int,
                           recipe: Recipe, question: str, verb_steps: List[str],
                           iter_type: str) -> str:
        """
        Creates intersection of steps in which there is a verb and semantic example
        """
        for example in semantic_role_examples.values():
            if example in question:
                if iter_type == "sentence":
                    example_steps = [idx[0] for idx, example_value in semantic_role_examples.items() if
                                     example_value == example]
                else:
                    example_steps = [idx[1] for idx, example_value in semantic_role_examples.items() if
                                     example_value == example]

                intersection = [value for value in verb_steps if value in example_steps]

                if intersection and intersection[0]:
                    answer = self.cut_rows_and_answer(recipe, intersection, iter_type, annotation_column)

                    if answer:
                        return answer
        return ""

    @staticmethod
    def search_drop_column(recipe: Recipe, verb: str, verb_steps: List[str], iter_type: str) -> str:
        """
        Get value from drop column for verb
        """
        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):

            part_identifier = sentence.sentence_id if iter_type == "sentence" else sentence.paragraph_id
            if part_identifier in verb_steps:
                for token in recipe.annotated_recipe.annotated_sentences[sentence_idx].annotated_tokens:
                    if token.normalized_token == verb:
                        return token.relation1
        return ""

    def make_use_of_drop(self, drop_column: str) -> str:
        drop_column = drop_column.split("|")[0].split("=")[1:][0].split(":")
        drop_column = [value.split(".")[0] for value in drop_column]
        drop_column = [" ".join(value.split("_")) for value in drop_column]
        drop_column = [value.replace(" - ", "-") for value in drop_column]
        drop_column = [self.inflection.singular_noun(value) if self.inflection.singular_noun(value) else value for value
                       in drop_column]
        drop_column = [value.replace("-", " - ") for value in drop_column]
        drop_column = ", ".join(drop_column)
        drop_column = drop_column.rsplit(",", 1)
        drop_column = " and".join(drop_column)
        return drop_column

    def verb_iteration(self, annotation_column: int, recipe: Recipe, question: str,
                       iter_type: str, verbs: Dict[tuple, str], semantic_role_examples: Dict[tuple, str] = {},
                       use_drop_column: bool = False, semantics: bool = True) -> str:
        """
        Iterating over all of the verbs in step
        """
        for verb in verbs.values():
            if verb in question:
                if iter_type == "sentence":
                    verb_steps = [idx[0] for idx, verb_value in verbs.items() if verb_value == verb]
                else:
                    verb_steps = [idx[1] for idx, verb_value in verbs.items() if verb_value == verb]
                if use_drop_column:
                    drop_column_value = self.search_drop_column(recipe, verb, verb_steps, iter_type)
                    if drop_column_value and "Drop" in drop_column_value:
                        drop_value = self.make_use_of_drop(drop_column_value)
                        if drop_value in question:
                            answer = self.cut_rows_and_answer(recipe, verb_steps, iter_type, annotation_column)
                            if answer:
                                return answer

                else:
                    if semantics:
                        answer = self.semantic_iteration(semantic_role_examples, annotation_column, recipe,
                                                         question, verb_steps, iter_type)
                        if answer:
                            return answer
                    else:
                        answer = self.cut_rows_and_answer(recipe, verb_steps, iter_type, annotation_column)
                        if answer:
                            return answer

        return ""

    def steps_and_columns_iteration(self, recipe: Recipe, question: str, paragraphs: List[str],
                                    iter_type: str, semantics: bool = True, semantic_roles: List[str] = None,
                                    use_drop_column: bool = False) -> str:
        """
        Iterate over paragraphs and columns
        """
        if not semantic_roles:
            semantic_roles = self.semantic_roles

        paragraphs = reversed(paragraphs) if self.reversed_paragraphs else paragraphs
        for paragraph in paragraphs:
            for annotation_column in range(0, 10):
                verbs = self.words_from_paragraph(recipe, paragraph, annotation_column, "V", False)
                if use_drop_column:
                    answer = self.verb_iteration(annotation_column, recipe, question, iter_type, verbs,
                                                 use_drop_column=True)
                    if answer:
                        return answer

                else:
                    if semantics:
                        for semantic_role in semantic_roles:
                            semantic_role_examples = self.words_from_paragraph(recipe, paragraph, annotation_column,
                                                                               semantic_role, True)
                            answer = self.verb_iteration(annotation_column, recipe, question, iter_type,
                                                         verbs, semantic_role_examples)
                            if answer:
                                return answer
                    else:
                        answer = self.verb_iteration(annotation_column, recipe, question, iter_type, verbs,
                                                     semantics=False)
                        if answer:
                            return answer

        return ""

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:

        """
        Answers the question "where do you". Iterate over paragraphs and semantic
        role's columns to find specified labels.
        :param question: question to be answered
        :param question_category: not checked, can be used for Location / Result / Cause / Purpose / Extent /
         Co-patient, etc.
        :param more_info: ignored
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswererUniversalSrl.DESCRIPTION}

        paragraphs = [sentence.paragraph_id for sentence in question.recipe.annotated_recipe.annotated_sentences]

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs, "sentence")
        more_info_for_answer["details_for_excel"] = "semantics || sentence"
        if answer:
            return PredictedAnswer(answer, raw_question=question.question,
                                   confidence=None, more_info=more_info_for_answer)

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs, "sentence",
                                                  semantics=False, use_drop_column=True)
        more_info_for_answer["details_for_excel"] = "relation || sentence"
        if answer:
            return PredictedAnswer(answer, raw_question=question.question,
                                   confidence=None, more_info=more_info_for_answer)

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs, "paragraph")
        more_info_for_answer["details_for_excel"] = "semantics || paragraph"
        if answer:
            return PredictedAnswer(answer, raw_question=question.question,
                                   confidence=None, more_info=more_info_for_answer)

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs, "paragraph",
                                                  semantics=False, use_drop_column=True)
        more_info_for_answer["details_for_excel"] = "relation || paragraph"
        if answer:
            return PredictedAnswer(answer, raw_question=question.question,
                                   confidence=None, more_info=more_info_for_answer)

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs, "sentence",
                                                  semantics=False, use_drop_column=False)
        more_info_for_answer["details_for_excel"] = "verb || sentence"
        if answer:
            return PredictedAnswer(answer, raw_question=question.question,
                                   confidence=None, more_info=more_info_for_answer)

        more_info_for_answer["details_for_excel"] = "no match"
        return PredictedAnswer(None, raw_question=question.question, confidence=None, more_info=more_info_for_answer)
