import io
from typing import Dict, Any, List

from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.putty_lemmatizer import PuttyLemmatizer
from src.annotated_recipe import AnnotatedToken, AnnotatedSentence
import inflect


class QuestionAnswererCountingTimes(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: HowManyTimes X is used?"

    def __init__(self):
        self.lemmatizer = PuttyLemmatizer()
        self.inflection_engine = inflect.engine()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: ignored
        :param more_info: "dump_logs_for_bad_answers" : True if you want the diagnostic logs
        :return: predicted answer
        """
        outstream = io.StringIO()

        print(f"Id = {question.recipe.id}", file=outstream)
        print(f"Q = {question.question}", file=outstream)
        the_object = self.get_object_from_question(question.question)
        print(f"Object = {the_object}", file=outstream)
        aliases = self.find_aliases(the_object, question)
        print(f"Aliases = {aliases}", file=outstream)

        last_found = []
        for alias in aliases:
            last_found.extend(self.search_for_coref_id(alias, question))

        print(f"Found = {last_found}", file=outstream)

        final_answer = str(len(last_found)) if last_found else None
        print(f"Final = {final_answer}", file=outstream)
        print(f"Truth = {question.answer}", file=outstream)

        details_str = f"Matched events = {last_found}"
        if (final_answer != question.answer and question.answer != "N/A") \
                or (question.answer == "N/A" and final_answer is not None):
            if more_info.get("dump_logs_for_bad_answers", False):
                print(outstream.getvalue())

        more_info_for_answer = {"source": QuestionAnswererCountingTimes.DESCRIPTION,
                                "details_for_excel": details_str}

        ret = PredictedAnswer(final_answer, more_info=more_info_for_answer)
        return ret

    def get_object_from_question(self, question: str) -> str:
        if question.find("How many times is the") != 0:
            return None
        as_array = question.lower().replace("?", " ").split(" ")
        objects = as_array[5:]
        objects = [o for o in objects if o and o != "used"]
        return "_".join(objects)

    def search_for_coref_id(self, alias: str, question: QuestionAnswerRecipe) -> List[AnnotatedToken]:
        if not alias:
            return []

        ret = []
        coref_id = ".".join(alias.split(".")[1:])

        for sentence in question.recipe.annotated_recipe.annotated_sentences:
            for token in sentence.annotated_tokens:
                multiplicity = self.search_for_duplication(sentence)

                found_in_rel2 = bool(token.relation2 and token.relation2.endswith(coref_id))
                if found_in_rel2:
                    ret.extend([token] * multiplicity)

                for r in self.get_valid_relaitons1(token):
                    if r.endswith(coref_id):
                        ret.extend([token] * multiplicity)

                # TODO handle duplications
        return ret

    def search_for_duplication(self, sentence: AnnotatedSentence) -> int:

        str_to_int = {
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5
        }

        scalar = 1
        for token in sentence.annotated_tokens:
            if token.part_of_speech != "NUM":
                continue
            semantic_roles = set(token.semantic_roles)
            allowed_roles = {"I-Theme", "B-Theme", "I-Location", "B-Location", "B-Destination", "B-Destination"}
            if semantic_roles.intersection(allowed_roles):
                scalar = str_to_int.get(token.normalized_token.lower(), 1)
        return scalar

    def get_valid_relaitons1(self, token: AnnotatedToken) -> List[str]:
        relations = token.get_whole_entry_from_relation1("Habitat") \
                    + token.get_whole_entry_from_relation1("Tool") \
                    + token.get_whole_entry_from_relation1("Drop") \
                    + token.get_whole_entry_from_relation1("Result") \
                    + token.get_whole_entry_from_relation1("Shadow")
        return relations

    def find_aliases(self, an_object: str, question: QuestionAnswerRecipe) -> List[str]:
        normalized = self.normalize_singular_plural_form(an_object)
        ret = []

        for normalized_phrase in normalized:
            objects_as_list = normalized_phrase.lower().split("_")
            l = len(objects_as_list)

            for sentence in question.recipe.annotated_recipe.annotated_sentences:
                for token in sentence.annotated_tokens:
                    i = token.id - 1
                    tokens = sentence.annotated_tokens[i:i + l] \
                        if i + l < len(sentence.annotated_tokens) else sentence.annotated_tokens[i:]
                    words = [t.raw_token.lower() for t in tokens]

                    if words == objects_as_list:
                        aliases = [t.relation2 for t in tokens if t.relation2]
                        aliases = [a for a in aliases if a]
                        for a in aliases:
                            if a not in ret:
                                ret.append(a)

                    for r in self.get_valid_relaitons1(token):
                        if r.startswith(normalized_phrase + ".") and r not in ret:
                            ret.append(r)
        return ret

    def normalize_singular_plural_form(self, phrase: str) -> List[str]:
        as_array = phrase.split("_")
        last_singular = self.inflection_engine.singular_noun(as_array[-1])
        last_singular = last_singular if last_singular else as_array[-1]
        last_plural = self.inflection_engine.plural_noun(as_array[-1])
        last_plural = last_plural if last_plural else as_array[-1]
        as_array[-1] = last_singular
        singular_alias = "_".join(as_array)
        as_array[-1] = last_plural
        plural_alias = "_".join(as_array)
        ret = []
        if singular_alias not in ret:
            ret.append(singular_alias)
        if plural_alias not in ret:
            ret.append(plural_alias)
        return ret
