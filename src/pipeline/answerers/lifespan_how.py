import re
from typing import Dict, Any, Tuple, List

import inflect
from pyinflect import getInflection

from src.annotated_recipe import AnnotatedRecipe
from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.putty_lemmatizer import PuttyLemmatizer


class AnswerClass:
    def __init__(self):
        self.verb = ""
        self.patients: List[str] = []
        self.habitat: List[str] = []
        self.tool: List[str] = []
        self.drop: List[str] = []
        self.shadow: List[str] = []


class QuestionAnswererLifespanHow(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer for class LifespanHow"

    def __init__(self):
        self.lemmatizer = PuttyLemmatizer()
        self.inflect_engine = inflect.engine()
        self.inflect_engine.classical()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        Answers the question "How did you get .*"

        :param question: question to be answered
        :param question_category: assumed to be lifespan how
        :param more_info: ignored
        :return: answer about contents
        """
        more_info_for_answer = {"source": QuestionAnswererLifespanHow.DESCRIPTION}
        answer = None

        sentences, verb_token = self.find_sentence_with_answer(question.question, question.recipe.annotated_recipe)

        if sentences:
            answers = []
            for pair in zip(sentences, verb_token):
                answers.append(self.retrieve_answer(pair[0], pair[1]))
            answer = answers[-1]

            if len(answer) < 13:
                answers.sort(reverse=True)
                answer = answers[0]

        return PredictedAnswer(answer, raw_question=question.question, confidence=None, more_info=more_info_for_answer)

    def find_sentence_with_answer(self, question: str, recipe: AnnotatedRecipe) -> Tuple[Any, Any]:
        """
        Finds in recipe a sentence which is related to the span
        """
        token_from_question = re.search(r'How did you get the (.+)\?', question).group(1)

        span_lemma, plural_span_lemma = self.get_spans_lists(token_from_question)
        sentences, verb_tokens = self.get_sentences_and_verbs(recipe, span_lemma, plural_span_lemma)

        if len(sentences) == 0 and ' and ' in question:
            token_from_question = re.search(r'.+ and (.+)', token_from_question).group(1)
            span_lemma, plural_span_lemma = self.get_spans_lists(token_from_question)
            sentences, verb_tokens = self.get_sentences_and_verbs(recipe, span_lemma, plural_span_lemma)

        if len(sentences) > 0:
            return sentences, verb_tokens

        return None, None

    def get_sentences_and_verbs(self, recipe: AnnotatedRecipe, span_lemma: List,
                                plural_span_lemma: List) -> Tuple[List, List]:
        sentences = []
        verb_tokens = []

        for sentence in recipe.annotated_sentences:
            if "ingredients" in sentence.sentence_id:
                continue

            for token in sentence.annotated_tokens:
                if token.relation1 is not None and "Result=" in token.relation1 and token.role_in_recipe == 'B-EVENT':
                    preprocessed_relation = token.get_entry_from_relation1("Result")
                    preprocessed_relation = preprocessed_relation[0].lower().rstrip("_.,")
                    relation_spans = preprocessed_relation.replace("_", " ")
                    relation_spans_lemma = self.lemmatize_span(relation_spans)
                    if span_lemma == relation_spans_lemma:
                        sentences.append(sentence)
                        verb_tokens.append(token)
                    elif plural_span_lemma == relation_spans_lemma:
                        sentences.append(sentence)
                        verb_tokens.append(token)

        return sentences, verb_tokens

    def get_spans_lists(self, token_from_question: str) -> Tuple[List, List]:
        span_lemma = self.lemmatize_span(token_from_question)

        plural_form = self.inflect_engine.plural_noun(token_from_question)
        plural_span_lemma = self.lemmatize_span(plural_form)

        return span_lemma, plural_span_lemma

    def lemmatize_span(self, token_from_question: str) -> List[str]:
        question_spans = token_from_question.split(" ")
        return [self.lemmatizer.lemmatize_noun(element) for element in question_spans]

    def retrieve_answer(self, sentence: Any, verb_token: Any) -> Any:
        """
        Composes an answer from the sentence content
        """
        answer = AnswerClass()

        patient_roles_in_recipe = ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"]
        habitat_roles_in_recipe = ["HABITAT"]
        habitat_roles = ['B-Theme', 'I-Theme', 'B-Instrument', 'I-Instrument', 'B-Location', 'I-Location',
                         'B-Destination', 'I-Destination', 'B-Co-Patient', 'I-Co-Patient', 'B-Patient', 'I-Patient',
                         'B-Source', 'I-Source']
        patient_roles = ['B-Patient', 'B-Theme', 'B-Co-Patient', 'B-Co-Theme', 'B-Result', 'B-Attribute',
                         'I-Patient', 'I-Theme', 'I-Co-Patient', 'I-Co-Theme', 'I-Result']
        tool_roles = ['B-Location', 'B-Destination', 'B-Instrument', 'B-Patient', 'B-Source', 'B-Theme',
                      'I-Patient', 'I-Location', 'I-Destination', 'I-Instrument', 'I-Source', 'I-Theme']

        if not sentence:
            return None

        answer.verb = self.get_inflection(verb_token.normalized_token.lower())
        verb_position = verb_token.id
        if "Habitat=" in verb_token.relation1:
            answer.habitat = verb_token.get_entry_from_relation1("Habitat")
        if "Tool=" in verb_token.relation1:
            answer.tool = verb_token.get_entry_from_relation1("Tool")
        if "Drop=" in verb_token.relation1:
            answer.drop = verb_token.get_entry_from_relation1("Drop")
        if "Shadow=" in verb_token.relation1:
            answer.shadow = verb_token.get_entry_from_relation1("Shadow")

        promising_tokens = []

        for token in sentence.annotated_tokens:
            if token.where_is_my_verb_explicit != verb_position:
                continue
            promising_tokens.append(token)
            set_token = set(token.semantic_roles)

            if set_token.intersection(set(patient_roles)) and answer.verb != "" \
                    and token.role_in_recipe[2:] in patient_roles_in_recipe:
                if token.role_in_recipe[:1] == "B":
                    answer.patients.append(self.get_elements(token, sentence, patient_roles_in_recipe))

            if set_token.intersection(set(habitat_roles)) and token.role_in_recipe[2:] in habitat_roles_in_recipe:
                if token.role_in_recipe[:1] == "B":
                    answer.habitat.append(self.get_elements(token, sentence, habitat_roles_in_recipe))

            if set_token.intersection(set(tool_roles)) and token.role_in_recipe[2:] == 'TOOL' \
                    and len(answer.tool) == 0 and token.part_of_speech != 'CCONJ':
                if token.relation2 is not None:
                    tool = re.search('(^.+[a-z])', token.relation2).group(1)
                    answer.tool.append(tool)
                else:
                    answer.tool.append(token.raw_token)

        if not answer.patients and not answer.drop and not answer.shadow:
            for token in promising_tokens:
                if answer.verb != "" and token.role_in_recipe[2:] in patient_roles_in_recipe:
                    if token.role_in_recipe[:1] == "B":
                        answer.patients.append(self.get_elements(token, sentence, patient_roles_in_recipe))

        full_answer = self.generate_answer(answer)

        if not full_answer:
            return None

        return full_answer

    @staticmethod
    def get_elements(token: Any, sentence: Any, patient_roles_in_recipe: List) -> str:
        token_id = next_id = token.id
        answer_element = token.raw_token
        if token_id < len(sentence.annotated_tokens) and token.role_in_recipe[2:] in patient_roles_in_recipe:
            for token in sentence.annotated_tokens[token.id:]:
                if token.role_in_recipe[2:] in patient_roles_in_recipe \
                        and next_id < len(sentence.annotated_tokens) and token.role_in_recipe[:1] == "I":
                    answer_element += f" {token.raw_token}"
                    next_id += 1
                else:
                    break
        return answer_element

    def generate_answer(self, answer: AnswerClass) -> Any:
        full_answer = "by "

        set_habitat = {"sheet", "board", "stove", "stovetop", "griddle", "surface", "counter", "plate", "mixer", "paper"}

        if answer.verb:
            full_answer += f"{answer.verb[0]} "
            if len(answer.verb) > 1:
                if len(answer.verb[1]) > len(answer.verb[0]):
                    full_answer += f"{answer.verb[1]} "
        if len(answer.drop) != 0:
            for element in answer.drop:
                try:
                    answer.patients.append(re.search(r'(^.+[a-z])', element).group(1))
                except AttributeError:
                    continue
        if len(answer.shadow) != 0 and len(answer.patients) == 0:
            answer.patients.append(answer.shadow[0])
        if len(answer.patients) != 0:
            if len(answer.patients) > 1:
                all_patients = ', '.join(patient for patient in answer.patients[:-1])
                all_patients += f" and {answer.patients[-1]}"
                full_answer += f"the {all_patients} "
            else:
                full_answer += f"the {answer.patients[0]} "
        if len(answer.habitat) != 0:
            singular_habitat = self.singular_element(answer.habitat[0].split("_"))
            set_answer_habitat = set(singular_habitat.split(" "))
            if answer.verb and answer.verb[0] in ['adding', 'returning', 'transferring']:
                full_answer += f"to the {singular_habitat} "
            elif answer.verb and answer.verb[0] in ['lifting', 'separating', 'obtaining', 'peeling']:
                full_answer += f"from the {singular_habitat} "
            elif set_answer_habitat.intersection(set_habitat):
                full_answer += f"on the {singular_habitat} "
            else:
                full_answer += f"in the {singular_habitat} "
        if len(answer.tool) != 0:
            singular_tool = self.singular_element(answer.tool[0].split("_"))
            full_answer += f"with the {singular_tool}"
        full_answer = full_answer.replace("_", " ").replace("/", " / ").replace(" '", "'").rstrip()
        full_answer = full_answer.lower()
        if " - " not in full_answer:
            full_answer = full_answer.replace("-", " - ")

        if full_answer == "by ":
            return None
        return full_answer

    def singular_element(self, phrase: Any) -> str:
        singular_tool = ""
        for element in phrase:
            plural = self.inflect_engine.singular_noun(element)
            singular_tool += element + " " if not plural else plural
        return singular_tool.rstrip()

    @staticmethod
    def get_inflection(verb_token: Any) -> List:
        irregular_inflection = {
            'saute': 'sauting',
            'sautee': 'sauteeing',
            'layering': 'layeringing',
            'dilut': 'diluting',
            'top': 'topping',
            'shre': 'shring',
            'kneed': 'kneeding',
            'golden': 'goldening brown',
            'tartar': 'tartarizing',
            'aging': 'aginging'
        }
        if verb_token in irregular_inflection.keys():
            for k, v in irregular_inflection.items():
                if verb_token == k:
                    return [v]
        else:
            return getInflection(verb_token, 'VBG', inflect_oov=True)
