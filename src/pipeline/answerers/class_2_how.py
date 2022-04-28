from src.pipeline.interface_question_answering import QuestionAnswerRecipe
from src.unpack_data import Recipe
from typing import Dict, List, Tuple
from src.annotated_recipe import AnnotatedSentence
import inflect
from src.putty_lemmatizer import PuttyLemmatizer


class QuestionAnswerer2How:
    DESCRIPTION = "QuestionAnswerer: How do you?"

    def __init__(self, semantic_roles: List[str], answer_annotations: List[str]):
        self.semantic_roles = semantic_roles
        self.answer_annotations = answer_annotations

        self.inflection = inflect.engine()
        self.lemmatizer = PuttyLemmatizer()

    def concat_words(self, sentence: AnnotatedSentence, annotation_column: int,
                     possible_answer: str, raw: bool = False, verb: bool = False) -> Dict[tuple, str]:
        """
        Concatenate words with the same semantic annotation (B-XX with I-XX)
        :param sentence: annotated sentence being analyzed
        :param annotation_column: column index where we look for semantic roles
        :param possible_answer: semantic role name we are looking for (like "V" in "B-V")
        :param raw: True if we want to return raw (not normalized) token
        :param verb: True if we want to search for verb (i.e. iterate over columns)
        :return: a dictionary indexed by (sentence id, paragraph id, token index) with corresponding words
        """

        words = {}
        extra_idx = None
        b_answer_found = False
        b_where_is_my_verb = None

        for token_idx, token in enumerate(sentence.annotated_tokens):
            value = token.raw_token.lower() if raw else token.normalized_token.lower()
            semantic_role = token.semantic_roles[annotation_column] if verb else token.role_in_recipe

            if semantic_role is None:
                continue

            # First we look for f"B-{possible_answer}
            if semantic_role == f"B-{possible_answer}":
                b_answer_found = True
                extra_idx = token.id
                b_where_is_my_verb = token.where_is_my_verb_explicit
                words[(sentence.sentence_id, sentence.paragraph_id, extra_idx, b_where_is_my_verb)] = [value]

            # If f"B-{possible_answer}" has been found, we append f"I-{possible_answer}"s if we find them
            elif b_answer_found and semantic_role == f"I-{possible_answer}":
                if self.inflection.singular_noun(value):
                    value = self.inflection.singular_noun(value)
                words[(sentence.sentence_id, sentence.paragraph_id, extra_idx, b_where_is_my_verb)].append(value)

        words = {key: " ".join(value) for key, value in words.items()}
        return words

    def words_from_paragraph(self, recipe: Recipe, paragraph: str, used_column: int,
                             searched_word: str, raw: bool, verb: bool) -> Dict[tuple, str]:
        """
        Collects all words with the same semantic annotation within the paragraph
        """

        words_in_paragraph = {}
        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            if sentence.paragraph_id == paragraph:
                words_in_paragraph.update(self.concat_words(sentence, used_column, searched_word, raw, verb))
                if verb:
                    words_in_paragraph.update(self.concat_words(sentence, used_column, "EVENT", raw))
        return words_in_paragraph

    def cut_rows_and_answer(self, verb, recipe: Recipe, steps: List[str], column: int,
                            answer_annotations: List[str] = None, verb_search: bool = False, verb_idx: int = 0) -> str:
        """
        Iterate over possible answer annotation and search for an answer
        """
        if not answer_annotations:
            answer_annotations = self.answer_annotations

        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            if sentence.sentence_id in steps:
                for annotation in answer_annotations:
                    answer_dict = self.concat_words(sentence, column, annotation, True, verb_search)
                    if answer_dict and list(answer_dict.keys())[0][3] == verb_idx:
                        answer = list(answer_dict.values())[0]
                        return answer

                tool_column_values = self.search_relation1_column(recipe, verb, steps)
                for tool_column_value in tool_column_values:
                    if tool_column_value and ("Tool" in tool_column_value):
                        tool_value = self.make_use_of_relation1("Tool", tool_column_value)
                        if tool_value:
                            return tool_value

        return ""

    def semantic_iteration(self, semantic_role_examples: Dict[tuple, str], used_column: int,
                           recipe: Recipe, question: str, verb_steps: List[str], verb: str,
                           verb_search: bool, verb_idx: int) -> str:
        """
        Creates intersection of steps in which there is a verb and semantic example
        """
        for example in semantic_role_examples.values():
            if example in question:
                example_steps = [idx[0] for idx, example_value in semantic_role_examples.items() if
                                 example_value == example if idx[3] == verb_idx]

                intersection = [value for value in verb_steps if value in example_steps]
                if intersection and intersection[0]:
                    answer = self.cut_rows_and_answer(verb, recipe, intersection, used_column,
                                                      verb_search=verb_search, verb_idx=verb_idx)

                    if answer:
                        return answer
        return ""

    @staticmethod
    def search_relation1_column(recipe: Recipe, verb: str, verb_steps: List[str]) -> List[str]:
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

    def make_use_of_relation1(self, relation: str, relation_column: str) -> str:
        relation_column = relation_column.split("|")
        relation_column = [value for value in relation_column if relation in value]
        relation_column = relation_column[0].split("=")[1:][0].split(":")
        relation_column = [value.split(".")[0] for value in relation_column]
        relation_column = [" ".join(value.split("_")) for value in relation_column]
        relation_column = ", ".join(relation_column)
        relation_column = relation_column.rsplit(",", 1)
        relation_column = " and".join(relation_column)
        relation_column = relation_column.replace(" '", "'")
        return relation_column

    def verb_iteration(self, used_column: int, recipe: Recipe, question: str, verbs: Dict[tuple, str],
                       semantic_role_examples: Dict[tuple, str] = {}, use_drop_column: bool = False,
                       semantics: bool = True) -> str:
        """
        Iterating over all of the verbs in step
        """
        for verb in verbs.values():
            if verb in question.split():
                verb_idx = [idx[2] for idx, verb_value in verbs.items() if verb_value == verb][0]
                verb_steps = [idx[0] for idx, verb_value in verbs.items() if verb_value == verb]

                if use_drop_column:
                    drop_column_values = self.search_relation1_column(recipe, verb, verb_steps)
                    for drop_column_value in drop_column_values:
                        if drop_column_value and ("Drop" in drop_column_value):
                            drop_value = self.make_use_of_relation1("Drop", drop_column_value)
                            if drop_value in question:
                                answer = self.cut_rows_and_answer(verb, recipe, verb_steps, used_column,
                                                                  verb_search=False, verb_idx=verb_idx)
                                if answer:
                                    return answer

                else:
                    if semantics:
                        answer = self.semantic_iteration(semantic_role_examples, used_column, recipe,
                                                         question, verb_steps, verb, False, verb_idx)
                        if answer:
                            return answer
                    else:
                        answer = self.cut_rows_and_answer(verb, recipe, verb_steps, used_column,
                                                          verb_idx=verb_idx)
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

                verbs = self.words_from_paragraph(recipe, paragraph, used_column, "V", False, True)
                if use_drop_column:
                    answer = self.verb_iteration(used_column, recipe, question, verbs, use_drop_column=True)
                    if answer:
                        return answer

                else:
                    if semantics:
                        for semantic_role in semantic_roles:
                            semantic_role_examples = self.words_from_paragraph(recipe, paragraph, used_column,
                                                                               semantic_role, True, False)
                            answer = self.verb_iteration(used_column, recipe, question, verbs, semantic_role_examples)
                            if answer:
                                return answer
                    else:
                        answer = self.verb_iteration(used_column, recipe, question, verbs, semantics=False)
                        if answer:
                            return answer

        return ""

    def final_answer(self, answer: str) -> str:
        if len(answer) <= 5 and any(word in answer for word in ["hand", "hands"]):
            return "by hand"
        return f"by using {self.lemmatizer.lemmatize_noun(answer)}"

    def answer_class2_question(self, question: QuestionAnswerRecipe) -> Tuple[str, Dict[str, str]]:

        """
        Answers the question "how do you", belonging to class 2. Iterate over paragraphs and semantic
        role's columns to find specified labels and relations.
        :param question: question to be answered
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswerer2How.DESCRIPTION}
        paragraphs = [sentence.paragraph_id for sentence in question.recipe.annotated_recipe.annotated_sentences]

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs)
        if answer:
            more_info_for_answer["details_for_excel"] = f"semantics || class 2"
            return self.final_answer(answer), more_info_for_answer

        answer = self.steps_and_columns_iteration(question.recipe, question.question.lower(), paragraphs,
                                                  semantics=False, use_drop_column=True)
        if answer:
            more_info_for_answer["details_for_excel"] = f"relation || class 2"
            return self.final_answer(answer), more_info_for_answer

        more_info_for_answer["details_for_excel"] = "no match || class 2"
        return "", more_info_for_answer
