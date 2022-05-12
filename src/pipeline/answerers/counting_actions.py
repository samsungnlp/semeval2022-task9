from typing import Dict, Any, List

import nltk
import io
from src.pipeline.interface_question_answering import QuestionAnsweringBase, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.unpack_data import QuestionAnswerRecipe
from src.putty_lemmatizer import PuttyLemmatizer
import inflect


def construct_map_with_i_and_h_columns(question: QuestionAnswerRecipe) -> Dict[str, int]:
    tools_and_habitats_map = {}
    for sentence in question.recipe.annotated_recipe.annotated_sentences:
        for token in sentence.annotated_tokens:
            words = []
            if token.relation1:
                words.extend(token.get_whole_entry_from_relation1("Drop"))
                words.extend(token.get_whole_entry_from_relation1("Tool"))
                words.extend(token.get_whole_entry_from_relation1("Habitat"))
                words.extend(token.get_whole_entry_from_relation1("Result"))
                words.extend(token.get_whole_entry_from_relation1("Shadow"))
            if token.relation2:
                words.append(token.relation2)

            for w in words:
                if w in tools_and_habitats_map:
                    tools_and_habitats_map[w] += 1
                else:
                    tools_and_habitats_map[w] = 1
    return tools_and_habitats_map


def find_occurrences(question_noun, tools_and_habitats_map):
    question_noun = question_noun.strip().replace(" ", "_")
    result = [(key, value) for key, value in tools_and_habitats_map.items() if
              nltk.edit_distance(key.split(".")[0], question_noun) < 2]
    return result


def count_raw_occurences(question_noun, question):
    count = 0
    for sentence in question.recipe.annotated_recipe.annotated_sentences:
        question_noun = question_noun.replace("_", " ")
        if question_noun in sentence.raw_sentence.lower() and "ingredients" not in sentence.sentence_id:
            count += sentence.raw_sentence.lower().count(question_noun)
    return count


def calculate_result(my_list: List) -> int:
    sum_of_all_occurrences = 0
    for r in my_list:
        sum_of_all_occurrences += r[1]
    return sum_of_all_occurrences


class QuestionAnswererCountingActions(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer How many actions"

    """
    Answers the question "How many actions does it take to... ?",
    :param question: question to be answered
    :param question_category: 
    :param more_info: ignored
    :return: count
    """

    def __init__(self):
        self.lemmatizer = PuttyLemmatizer()
        self.inflect_engine = inflect.engine()
        self.inflect_engine.classical()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:

        outstream = io.StringIO()
        print(f"Id = {question.recipe.id} || {question.question_class}", file=outstream)
        print(f"Q = {question.question}", file=outstream)
        the_object = self.get_object_from_question(question)
        print(f"Object = {the_object}", file=outstream)

        relations_map = construct_map_with_i_and_h_columns(question)
        list_singular = find_occurrences(the_object, relations_map)
        print(f"Singular = {list_singular}", file=outstream)

        object_as_plural = self.inflect_engine.plural_noun(the_object)
        print(f"Plural = {object_as_plural}", file=outstream)
        list_plural = find_occurrences(object_as_plural, relations_map)
        print(f"Plural    = {list_plural}", file=outstream)

        result_singular = calculate_result(list_singular)
        result_plural = calculate_result(list_plural)
        final_answer = max(result_singular, result_plural)
        print(f"Trying from relation match = {final_answer}", file=outstream)
        last_rule = "Relation match"

        if not final_answer:
            final_answer = count_raw_occurences(the_object, question)
            last_rule = "Raw occurences"
            print(f"Trying raw occurences = {final_answer}", file=outstream)

        final_answer = str(final_answer) if final_answer else None
        if not final_answer:
            last_rule = "Nothing found"

        print(f"Final = {final_answer}", file=outstream)
        print(f"Truth = {question.answer}\n", file=outstream)

        details_str = f"Last rule = {last_rule}"
        if (final_answer != question.answer and question.answer != "N/A") \
                or (question.answer == "N/A" and final_answer is not None):
            if more_info.get("dump_logs_for_bad_answers", False):
                print(outstream.getvalue())

        more_info_for_answer = {"source": QuestionAnswererCountingActions.DESCRIPTION,
                                "details_for_excel": details_str}

        return PredictedAnswer(final_answer, raw_question=question.question, confidence=None,
                               more_info=more_info_for_answer)

    def get_object_from_question(self, question: QuestionAnswerRecipe) -> str:
        q = question.question.replace("?", "").lower().strip().split(" ")[9:]
        return " ".join(q)
