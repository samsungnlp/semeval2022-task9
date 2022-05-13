from typing import Dict, List, Any, Tuple

import inflect
import nltk

from src.annotated_recipe import AnnotatedSentence
from src.pipeline.interface_question_answering import QuestionAnswerRecipe, QuestionAnsweringBase, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import Recipe


class QuestionAnswererEllipsisV2(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: What should be?"

    def __init__(self, semantic_roles: List[str], answer_annotations: List[str], answer_relations: List[str]):
        self.semantic_roles = semantic_roles
        self.answer_annotations = answer_annotations
        self.answer_relations = answer_relations

        self.inflection = inflect.engine()

    @staticmethod
    def concat_words(sentence: AnnotatedSentence, annotation_column: int, possible_answer: str, raw: bool = False,
                     verb_search: bool = False, singular: bool = False) -> Dict[tuple, str]:
        """
        Concatenate words with the same semantic annotation (B-XX with I-XX)
        :param sentence: annotated sentence being analyzed
        :param annotation_column: column index where we look for semantic roles
        :param possible_answer: semantic role name we are looking for (like "V" in "B-V")
        :param raw: True if we want to return raw (not normalized) token
        :param verb_search: True if we want to search for verb (i.e. iterate over columns)
        :param singular: True if we want to search for one semantic label at the time
        :return: a dictionary indexed by (sentence id, paragraph id, token index) with corresponding words
        """

        words = {}
        extra_idx = None
        b_answer_found = False
        ingredient_verb_token_idx = None
        ingredient_semantic_roles = None

        for token_idx, token in enumerate(sentence.annotated_tokens):
            value = token.raw_token.lower() if raw else token.normalized_token.lower()

            semantic_role = token.semantic_roles[annotation_column] if verb_search else token.role_in_recipe

            if semantic_role is None:
                continue

            searched_annotations_b = [f"B-{possible_answer}"] if singular \
                else ["B-EXPLICITINGREDIENT", "B-IMPLICITINGREDIENT"]
            searched_annotations_i = [f"I-{possible_answer}"] if singular \
                else ["I-EXPLICITINGREDIENT", "I-IMPLICITINGREDIENT"]

            # First we look for f"B-{possible_answer}
            if semantic_role in searched_annotations_b:
                b_answer_found = True
                extra_idx = token.id
                ingredient_verb_token_idx = token.where_is_my_verb_explicit
                ingredient_semantic_roles = tuple(token.semantic_roles)
                words[(sentence.sentence_id, sentence.paragraph_id, extra_idx, ingredient_verb_token_idx, ingredient_semantic_roles)] = [value]

            # If f"B-{possible_answer}" has been found, we append f"I-{possible_answer}"s if we find them
            elif b_answer_found and semantic_role in searched_annotations_i:
                words[(sentence.sentence_id, sentence.paragraph_id, extra_idx, ingredient_verb_token_idx, ingredient_semantic_roles)].append(value)

        words = {key: " ".join(value) for key, value in words.items()}
        return words

    def words_from_paragraph(self, recipe: Recipe, paragraph: str, used_column: int,
                             searched_word: str, raw: bool, verb_search: bool, singular: bool) -> Dict[tuple, str]:
        """
        Collects all words with the same semantic annotation within the paragraph
        """

        words_in_paragraph = {}
        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            if sentence.sentence_id == paragraph:
                words_in_paragraph.update(self.concat_words(
                    sentence=sentence,
                    annotation_column=used_column,
                    possible_answer=searched_word,
                    raw=raw,
                    verb_search=verb_search,
                    singular=singular
                ))
                if verb_search:
                    words_in_paragraph.update(self.concat_words(
                        sentence=sentence,
                        annotation_column=used_column,
                        possible_answer="EVENT",
                        raw=raw,
                        singular=True
                    ))
        return words_in_paragraph

    def cut_rows_and_answer(self, verb, recipe: Recipe, steps: List[str], iter_type: str, answer_relations: str = "",
                            ingredients: Tuple[str, str] = ("", "")) -> str:
        """
        Iterate over possible answer annotation and search for an answer
        """
        if not answer_relations:
            answer_relations = self.answer_relations

        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):

            part_identifier = sentence.sentence_id if iter_type == "sentence" else sentence.paragraph_id

            if part_identifier in steps:
                relation1_column_values = self.search_relation1_column(
                    recipe=recipe,
                    verb=verb,
                    verb_steps=steps,
                    iter_type=iter_type
                )
                for relation1_column_value in relation1_column_values:
                    for answer_relation in answer_relations:
                        if relation1_column_value and (answer_relation in relation1_column_value):
                            relation1_value = self.make_use_of_relation1(
                                relation=answer_relation,
                                relation_column_value=relation1_column_value,
                                make_singular=False
                            )

                            if relation1_value:
                                if ingredients and ingredients[1] == sentence.sentence_id:
                                    if ingredients[0] not in relation1_value:
                                        answer = f"{ingredients[0]}, {relation1_value}"
                                        answer = answer.rsplit(",", 1)
                                        answer = " and".join(answer)
                                        return f"the {answer}"
                                relation1_value = relation1_value.rsplit(",", 1)
                                relation1_value = " and".join(relation1_value)
                                return f"the {relation1_value}"

        return ""

    def semantic_iteration(self, semantic_role_examples: Dict[tuple, str], recipe: Recipe, question: str,
                           verb_steps: List[str], iter_type: str, verb: str, ingredients: Tuple[str, str]) -> str:
        """
        Creates intersection of steps in which there is a verb and semantic example
        """
        for example in semantic_role_examples.values():
            example_singular = example.replace(" - ", "-")
            example_singular = self.inflection.singular_noun(example_singular)
            example_singular = example_singular.replace("-", " - ") if example_singular else example

            if example in question or example_singular in question:
                if iter_type == "sentence":
                    example_steps = [idx[0] for idx, example_value in semantic_role_examples.items() if
                                     example_value == example]
                else:
                    example_steps = [idx[1] for idx, example_value in semantic_role_examples.items() if
                                     example_value == example]

                intersection = [value for value in verb_steps if value in example_steps]
                if intersection and intersection[0]:
                    answer = self.cut_rows_and_answer(
                        verb=verb,
                        recipe=recipe,
                        steps=intersection,
                        iter_type=iter_type,
                        ingredients=ingredients
                    )

                    if answer:
                        return answer
        return ""

    @staticmethod
    def search_relation1_column(recipe: Recipe, verb: str, verb_steps: List[str], iter_type: str) -> List[str]:
        """
        Get value from drop column for verb
        """
        relations = []
        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            part_identifier = sentence.sentence_id if iter_type == "sentence" else sentence.paragraph_id
            if part_identifier in verb_steps:
                for token in recipe.annotated_recipe.annotated_sentences[sentence_idx].annotated_tokens:
                    if token.normalized_token.lower() in verb and token.relation1:
                        relations.append(token.relation1)
        return relations

    def make_use_of_relation1(self, relation: str, relation_column_value: str, make_singular: bool) -> str:
        relation_column_value = relation_column_value.split("|")
        relation_column_value = [value for value in relation_column_value if relation in value]
        relation_column_value = relation_column_value[0].split("=")[1:][0].split(":")
        relation_column_value = [value.split(".")[0] for value in relation_column_value]
        relation_column_value = [" ".join(value.split("_")) for value in relation_column_value]
        if make_singular:
            relation_column_value = [value.replace(" - ", "-") for value in relation_column_value]
            relation_column_value = [self.inflection.singular_noun(value) if self.inflection.singular_noun(value)
                                     else value for value in relation_column_value]
            relation_column_value = [value.replace("-", " - ") for value in relation_column_value]
        relation_column_value = ", ".join(relation_column_value)
        return relation_column_value.lower()

    def get_ingredients(self, recipe: Recipe, verb: str, verbs: Dict[tuple, str],
                        answer_annotations: List[str] = None) -> Tuple[str, str]:
        if not answer_annotations:
            answer_annotations = self.answer_annotations

        forbidden_semantic_roles = {"B-Result", "I-Result", "B-Location", "I-Location",
                                    "B-Destination", "I-Destination", "B-Attribute", "I-Attribute",
                                    "B-Material", "I-Material", "B-Instrument", "I-Instrument",
                                    "B-Source", "I-Source", "B-Time", "I-Time", "B-Product", "I-Product"}
        verb_steps = [idx[0] for idx, verb_value in verbs.items() if verb_value == verb]
        verb_idx = [idx[2] for idx, verb_value in verbs.items() if verb_value == verb][0]

        for sentence_idx, sentence in enumerate(recipe.annotated_recipe.annotated_sentences):
            for verb_step in verb_steps:
                if sentence.sentence_id in verb_step:
                    for annotation in answer_annotations:
                        answer_dict = self.concat_words(
                            sentence=sentence,
                            possible_answer=annotation,
                            annotation_column=0,
                            raw=True,
                            verb_search=False
                        )

                        ingredients = [value for idx, value in answer_dict.items() if idx[3] == verb_idx
                                       and set(idx[4]) - forbidden_semantic_roles == set(idx[4])]
                        if ingredients:
                            return ", ".join(ingredients), verb_step
        return "", ""

    def verb_iteration(self, used_column: int, recipe: Recipe, question: str,
                       iter_type: str, verbs: Dict[tuple, str], semantic_role_examples: Dict[tuple, str] = {},
                       use_drop_column: bool = False, semantics: bool = True, searched_relation1_values: List[str] = [""]) -> str:
        """
        Iterating over all of the verbs in step
        """

        for verb in verbs.values():
            if verb in question:
                if iter_type == "sentence":
                    verb_steps = [idx[0] for idx, verb_value in verbs.items() if verb_value == verb]
                else:
                    verb_steps = [idx[1] for idx, verb_value in verbs.items() if verb_value == verb]

                ingredients = self.get_ingredients(
                    recipe=recipe,
                    verb=verb,
                    verbs=verbs
                )

                if use_drop_column:
                    relation1_column_values = self.search_relation1_column(
                        recipe=recipe,
                        verb=verb,
                        verb_steps=verb_steps,
                        iter_type=iter_type
                    )
                    for relation1_column_value in relation1_column_values:
                        if relation1_column_value:
                            for searched_relation1_value in searched_relation1_values:
                                if searched_relation1_value in relation1_column_value:
                                    relation1_value = self.make_use_of_relation1(
                                        relation=searched_relation1_value,
                                        relation_column_value=relation1_column_value,
                                        make_singular=True
                                    )

                                    if relation1_value in question:
                                        answer = self.cut_rows_and_answer(
                                            verb=verb,
                                            recipe=recipe,
                                            steps=verb_steps,
                                            iter_type=iter_type,
                                            ingredients=ingredients
                                        )
                                        if answer:
                                            return answer

                else:
                    if semantics:
                        answer = self.semantic_iteration(
                            semantic_role_examples=semantic_role_examples,
                            recipe=recipe,
                            question=question,
                            verb_steps=verb_steps,
                            iter_type=iter_type,
                            verb=verb,
                            ingredients=ingredients
                        )
                        if answer:
                            return answer
                    else:
                        answer = self.cut_rows_and_answer(
                            verb=verb,
                            recipe=recipe,
                            steps=verb_steps,
                            iter_type=iter_type,
                            ingredients=ingredients
                        )
                        if answer:
                            return answer

        return ""

    def steps_and_columns_iteration(self, recipe: Recipe, question: str, paragraphs: List[str], iter_type: str,
                                    searched_relation1_values: List[str] = [""], semantics: bool = True,
                                    semantic_roles: List[str] = None, use_drop_column: bool = False) -> List[str]:
        """
        Iterate over paragraphs and columns

        :param recipe:
        :param question:
        :param paragraphs:
        :param iter_type:
        :param semantics:
        :param semantic_roles:
        :param use_drop_column:
        :return:
        """
        if not semantic_roles:
            semantic_roles = self.semantic_roles

        answers = []
        for paragraph in paragraphs:
            for used_column in range(0, 10):

                verbs = self.words_from_paragraph(
                    recipe=recipe,
                    paragraph=paragraph,
                    used_column=used_column,
                    searched_word="V",
                    raw=False,
                    verb_search=True,
                    singular=True
                )

                if use_drop_column:
                    answer = self.verb_iteration(
                        used_column=used_column,
                        recipe=recipe,
                        question=question,
                        iter_type=iter_type,
                        verbs=verbs,
                        use_drop_column=True,
                        searched_relation1_values=searched_relation1_values
                    )
                    if answer and answer not in answers:
                        answers.append(answer)

                else:
                    if semantics:
                        for semantic_role in semantic_roles:
                            semantic_role_examples = self.words_from_paragraph(
                                recipe=recipe,
                                paragraph=paragraph,
                                used_column=used_column,
                                searched_word=semantic_role,
                                raw=True,
                                verb_search=False,
                                singular=True
                            )
                            answer = self.verb_iteration(
                                used_column=used_column,
                                recipe=recipe,
                                question=question,
                                iter_type=iter_type,
                                verbs=verbs,
                                semantic_role_examples=semantic_role_examples
                            )
                            if answer and answer not in answers:
                                answers.append(answer)
                    else:
                        answer = self.verb_iteration(
                            used_column=used_column,
                            recipe=recipe,
                            question=question,
                            iter_type=iter_type,
                            verbs=verbs,
                            semantics=False
                        )
                        if answer and answer not in answers:
                            answers.append(answer)

        return answers

    @staticmethod
    def change_verb_from_participle(question: str) -> str:
        """
        Changes verb in question from participle form
        """
        question = question.replace("?", "")
        question = question.split()
        question[3] = nltk.stem.WordNetLemmatizer().lemmatize(question[3], pos='v')
        question = " ".join(question)
        return question

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        Answers the question "what should be", belonging to ellipsis. Approach 2
        Iterate over paragraphs and semantic
        role's columns to find specified labels.
        :param question: question to be answered
        :param question_category: assumed to be ellipsis
        :param more_info: ignored
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswererEllipsisV2.DESCRIPTION}

        paragraphs = [sentence.sentence_id for sentence in question.recipe.annotated_recipe.annotated_sentences]

        question_changed = self.change_verb_from_participle(question.question.lower()) \
            if "plated" not in question.question else question.question.lower()
        question_changed = question_changed.replace("goldened", "golden") \
            if "goldened" in question_changed else question_changed
        question_changed = question_changed.replace("blent", "blend") \
            if "blent" in question_changed else question_changed
        question_changed = question_changed.replace("plated", "plate") \
            if "plated" in question_changed else question_changed

        all_answers = []

        if any(preposition in question_changed for preposition in ["in the", "from the", "to the", "on the"]):
            answers_relation = self.steps_and_columns_iteration(
                recipe=question.recipe,
                question=question_changed,
                paragraphs=paragraphs,
                iter_type="sentence",
                semantics=False,
                use_drop_column=True,
                searched_relation1_values=["Habitat"]
            )
            all_answers = all_answers+answers_relation

            answers_semantics = self.steps_and_columns_iteration(
                recipe=question.recipe,
                question=question_changed,
                paragraphs=paragraphs,
                iter_type="sentence",
                semantics=True,
                semantic_roles=["HABITAT"]
            )

            all_answers = all_answers + answers_semantics
            if all_answers:
                more_info_for_answer["details_for_excel"] = \
                    f"habitat object || relation: {answers_relation} || semantics: {answers_semantics}"
                return PredictedAnswer(all_answers[0], raw_question=question.question,
                                       confidence=None, more_info=more_info_for_answer)

        if "with the" in question_changed:
            answers_relation = self.steps_and_columns_iteration(
                recipe=question.recipe,
                question=question_changed,
                paragraphs=paragraphs,
                iter_type="sentence",
                semantics=False,
                use_drop_column=True,
                searched_relation1_values=["Tool"]
            )
            all_answers = all_answers + answers_relation

            answers_semantics = self.steps_and_columns_iteration(
                recipe=question.recipe,
                question=question_changed,
                paragraphs=paragraphs,
                iter_type="sentence",
                semantics=True,
                semantic_roles=["TOOL"]
            )
            all_answers = all_answers + answers_semantics

            if all_answers:
                more_info_for_answer["details_for_excel"] = \
                    f"tool object || relation: {answers_relation} || semantics: {answers_semantics}"
                return PredictedAnswer(all_answers[0], raw_question=question.question,
                                       confidence=None, more_info=more_info_for_answer)

        elif len(question_changed.split()) == 4 or (len(question_changed.split()) == 5
                                                    and question_changed.split()[4] == "brown"):
            answers = self.steps_and_columns_iteration(
                recipe=question.recipe,
                question=question_changed,
                paragraphs=paragraphs,
                iter_type="sentence",
                semantics=False,
            )
            if answers:
                more_info_for_answer[
                    "details_for_excel"] = f"no object || verb: {answers}"
                return PredictedAnswer(answers[0], raw_question=question.question,
                                       confidence=None, more_info=more_info_for_answer)

        more_info_for_answer["details_for_excel"] = "no match"
        return PredictedAnswer(None, raw_question=question.question, confidence=None, more_info=more_info_for_answer)
