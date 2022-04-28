from typing import List, Set

from src.putty_lemmatizer import PuttyLemmatizer
from src.annotated_recipe import AnnotatedSentence, AnnotatedToken
from src.unpack_data import QuestionAnswerRecipe


class VerbPatientHabitat:
    lemmatizer = PuttyLemmatizer()

    def __init__(self, verb: str, patients: List[str], habitats: List[str] = [], type=None, sentence_id: int = None,
                 token_id: int = None, all_related_words: List[str] = None):
        self.verb = verb
        self.patients = [patient.split(".")[0].lower() for patient in patients]
        self.habitats = [h for h in habitats if h]
        self.sentence_id = sentence_id
        self.all_related_words: List[str] = all_related_words
        self.token_id = token_id
        self.type = type

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.verb} {self.patients} H:{self.habitats}"

    def get_common_patients(self, other_event: "VerbPatientHabitat") -> Set[str]:
        my_patients = set(self.patients)
        other_patients = set(other_event.patients)
        return my_patients.intersection(other_patients)

    def is_exact_match_to_any_patients(self, word: str) -> bool:
        return word in self.patients

    def is_soft_match_to_any_patients(self, word: str) -> bool:
        return any(word in patient for patient in self.patients)

    @staticmethod
    def build_list_of_events(question: QuestionAnswerRecipe) -> "List[VerbPatientHabitat]":
        vphs: List[VerbPatientHabitat] = []

        for i, s in enumerate(question.recipe.annotated_recipe.annotated_sentences):
            VerbPatientHabitat.__append_vphs_from_sentence(s, vphs, i)
        return vphs

    @staticmethod
    def build_list_of_events_for_c17(question: QuestionAnswerRecipe) -> "List[VerbPatientHabitat]":
        vphs: List[VerbPatientHabitat] = []

        for i, s in enumerate(question.recipe.annotated_recipe.annotated_sentences):
            VerbPatientHabitat.__append_vphs_from_sentence_for_c17(s, vphs, i)
        return vphs

    @staticmethod
    def is_valid_event_verb(verb_token: AnnotatedToken) -> bool:
        return "B-V" in verb_token.semantic_roles \
               or (verb_token.role_in_recipe == "B-EVENT" and (
                "D-V" in verb_token.semantic_roles or "I-V" in verb_token.semantic_roles)) \
               or (verb_token.role_in_recipe == "B-EVENT" and verb_token.relation1 is not None)

    @staticmethod
    def is_valid_event_verb_17(verb_token: AnnotatedToken) -> bool:
        return "B-V" in verb_token.semantic_roles \
               or (verb_token.role_in_recipe == "B-EVENT" and (
                "D-V" in verb_token.semantic_roles or "I-V" in verb_token.semantic_roles))

    @staticmethod
    def __append_vphs_from_sentence(sentence: AnnotatedSentence, vphs: "List[VerbPatientHabitat]",
                                    current_sentence: int):
        for current_token, verb_token in enumerate(sentence.annotated_tokens):
            if not VerbPatientHabitat.is_valid_event_verb(verb_token):
                continue

            verb = verb_token.normalized_token.lower()
            verb = VerbPatientHabitat.lemmatizer.lemmatize_verb(verb)
            objects = VerbPatientHabitat.get_objects(verb_token, sentence)
            habitats = VerbPatientHabitat.get_habitats(verb_token, sentence)
            all_related_words = VerbPatientHabitat.get_all_related_words(verb_token, sentence)

            vphs.append(VerbPatientHabitat(verb, objects, habitats, sentence_id=current_sentence,
                                           token_id=current_token, all_related_words=all_related_words))

    @staticmethod
    def __append_vphs_from_sentence_for_c17(sentence: AnnotatedSentence, vphs: "List[VerbPatientHabitat]",
                                            current_sentence: int):
        for current_token, verb_token in enumerate(sentence.annotated_tokens):
            if not VerbPatientHabitat.is_valid_event_verb_17(verb_token):
                continue

            verb = verb_token.normalized_token.lower()
            verb = VerbPatientHabitat.lemmatizer.lemmatize_verb(verb)
            objects = VerbPatientHabitat.get_objects(verb_token, sentence)
            habitats = VerbPatientHabitat.get_habitats_for_c17(verb_token, sentence, also_add_from_raw_tokens=True)
            all_related_words = VerbPatientHabitat.get_all_related_words(verb_token, sentence)

            vphs.append(VerbPatientHabitat(verb, objects, habitats, sentence_id=current_sentence,
                                           token_id=current_token, all_related_words=all_related_words))

    @staticmethod
    def get_all_related_words(verb_token: AnnotatedToken, sentence: AnnotatedSentence) -> List[str]:
        verb_position = verb_token.id
        verb_semantic_role_ids = [i for i, sr in enumerate(verb_token.semantic_roles) if sr in ["B-V", "D-V"]]

        words = []
        for token in sentence.annotated_tokens:
            if token.is_equal_to_any_verb_id(verb_position) or \
                    any(token.semantic_roles[i] for i in verb_semantic_role_ids):
                words.append(token.raw_token.lower())
        return words

    @staticmethod
    def get_objects(verb_token: AnnotatedToken, sentence: AnnotatedSentence) -> List[str]:
        objects = verb_token.get_entry_from_relation1("Drop") + verb_token.get_entry_from_relation1("Result")
        for token in sentence.annotated_tokens:
            if token.relation2 and \
                    token.role_in_recipe in ["B-EXPLICITINGREDIENT", "B-IMPLICITINGREDIENT"] and \
                    token.is_equal_to_any_verb_id(verb_token.id):
                obj = token.relation2.split(".")[0]
                if obj not in objects:
                    objects.append(obj.lower())
                if token.normalized_token not in objects:
                    objects.append(token.normalized_token.lower())
        objects = VerbPatientHabitat.lemmatize_patients(objects)
        return objects

    @staticmethod
    def get_habitats(verb_token: AnnotatedToken, sentence: AnnotatedSentence, also_add_from_raw_tokens: bool = True) -> \
            List[str]:
        habitats = []
        for habitat_token in sentence.annotated_tokens:
            if habitat_token.role_in_recipe in {"B-HABITAT", "I-HABITAT"} and \
                    habitat_token.is_equal_to_any_verb_id(verb_token.id) and \
                    all([x not in habitat_token.semantic_roles for x in {"I-Patient", "B-Patient"}]):

                if also_add_from_raw_tokens and habitat_token.role_in_recipe == "B-HABITAT":
                    raw_habitat = VerbPatientHabitat.get_habitat_from_raw_token(habitat_token, sentence)
                    if raw_habitat not in habitats:
                        habitats.append(raw_habitat)
                if habitat_token.relation2:
                    hab = habitat_token.relation2.split(".")[0].lower()
                    if hab not in habitats:
                        habitats.append(hab)
        habitats.extend(verb_token.get_entry_from_relation1("Habitat"))
        return habitats

    @staticmethod
    def get_habitats_for_c17(verb_token: AnnotatedToken, sentence: AnnotatedSentence,
                             also_add_from_raw_tokens: bool = True) -> \
            List[str]:
        habitats = []
        for habitat_token in sentence.annotated_tokens:
            if habitat_token.role_in_recipe in {"B-HABITAT", "I-HABITAT"} and \
                    habitat_token.is_equal_to_any_verb_id(verb_token.id) and \
                    all([x not in habitat_token.semantic_roles for x in {"I-Patient", "B-Patient"}]):

                if also_add_from_raw_tokens and habitat_token.role_in_recipe == "B-HABITAT":
                    raw_habitat = VerbPatientHabitat.get_habitat_from_raw_token(habitat_token, sentence)
                    if habitat_token.relation2:
                        pos = habitat_token.relation2.find(".")
                        raw_habitat += habitat_token.relation2[pos:]
                    if raw_habitat not in habitats:
                        habitats.append(raw_habitat)
                if habitat_token.relation2:
                    hab = habitat_token.relation2
                    if hab not in habitats:
                        habitats.append(hab)
        habitats.extend(verb_token.get_whole_entry_from_relation1("Habitat"))
        return habitats

    @staticmethod
    def get_habitat_from_raw_token(habitat_token: AnnotatedToken, sentence: AnnotatedSentence) -> str:
        res = []
        t = habitat_token
        while t and t.role_in_recipe in {"B-HABITAT", "I-HABITAT"}:
            res.append(t.raw_token.lower())
            # Note: tokens are indexed from 1, so the next token is at annotated_token[current_token_id]
            t = sentence.annotated_tokens[t.id] if t.id < len(sentence.annotated_tokens) else None
        raw_habitat = "_".join(res)
        return raw_habitat

    @staticmethod
    def lemmatize_patients(patients: List[str]) -> List[str]:

        ret = []
        for patient in patients:
            source = patient.split(".")[0].lower()
            ret.append(source)
            as_array = source.split("_")
            last_word = as_array[-1]
            normalized_last = VerbPatientHabitat.lemmatizer.lemmatize_noun(last_word)
            as_array[-1] = normalized_last
            renormalized = "_".join(as_array)
            if renormalized != source:
                ret.append(renormalized)
        return ret
