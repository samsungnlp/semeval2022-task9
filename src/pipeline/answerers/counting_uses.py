import re
from typing import Dict, Any

import nltk
import io

from src.pipeline.interface_question_answering import QuestionAnsweringBase, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.putty_lemmatizer import PuttyLemmatizer
from src.unpack_data import QuestionAnswerRecipe
import inflect


def constuct_map_with_i_and_h_columns_tools(question: QuestionAnswerRecipe):
    return construct_map(question, "Tool", "TOOL")


def constuct_map_with_i_and_h_columns_habitats(question: QuestionAnswerRecipe):
    return construct_map(question, "Habitat", "HABITAT")


def construct_map(question: QuestionAnswerRecipe, rel1: str, role_in_recipe: str) -> Dict[str, int]:
    ret = {}
    for sentence in question.recipe.annotated_recipe.annotated_sentences:
        for token in sentence.annotated_tokens:
            words = []
            if token.relation1:
                words.extend(token.get_whole_entry_from_relation1(rel1))

            if token.relation2 and role_in_recipe in token.role_in_recipe:
                words.extend(re.split('[:;|=]', token.relation2))

            for w in words:
                if w in ret:
                    ret[w] += 1
                else:
                    ret[w] = 1
    return ret


def find_occurrences(question_noun, tools_and_habitats_map):
    question_noun = question_noun.strip().replace(" ", "_")
    result = [(key, value) for key, value in tools_and_habitats_map.items() if
              nltk.edit_distance(key.split(".")[0], question_noun) <= 1]
    return result


def count_raw_occurences(question_noun, question):
    count = 0
    for sentence in question.recipe.annotated_recipe.annotated_sentences:
        question_noun = question_noun.replace("_", " ")
        if question_noun in sentence.raw_sentence.lower() and "ingredients" not in sentence.sentence_id:
            count += sentence.raw_sentence.lower().count(question_noun)
    return count


def calculate_result(actions_or_times, my_list):
    if not my_list:
        result = 0
    elif actions_or_times:
        sum_of_all_occurrences = 0
        for r in my_list:
            sum_of_all_occurrences += r[1]
        result = sum_of_all_occurrences
    else:
        result = len(my_list)
    return result


class QuestionAnswererCountingUses(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer counting things"

    """
    Answers the question  "How many X are used?" 
    :param question: question to be answered
    :param question_category: assumed to be Counting uses
    :param more_info: ignored
    :return: count
    """

    def __init__(self):
        self.inflect_engine = inflect.engine()
        self.inflect_engine.classical()
        self.lemmatizer = PuttyLemmatizer()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:

        actions_or_times = False

        outstream = io.StringIO()
        print(f"Id = {question.recipe.id} || {question.question_class}", file=outstream)
        print(f"Q = {question.question}", file=outstream)
        singular_object = self.inflect_engine.singular_noun(self.get_object_from_question(question))
        plural_object = self.inflect_engine.plural_noun(singular_object)
        print(f"Object = {singular_object} // {plural_object}", file=outstream)

        # question_noun = question_noun.replace('``', "\"").lower().strip()
        last_rule = "Exact match"
        tools_map = constuct_map_with_i_and_h_columns_tools(question)
        habitats_map = constuct_map_with_i_and_h_columns_habitats(question)

        print(f"Tools = {tools_map}", file=outstream)
        print(f"Habitats = {habitats_map}", file=outstream)

        list_singular_tools = find_occurrences(singular_object, tools_map)
        list_plural_tools = find_occurrences(plural_object, tools_map)
        result_singular_tools = calculate_result(actions_or_times, list_singular_tools)
        result_plural_tools = calculate_result(actions_or_times, list_plural_tools)

        list_singular_habitats = find_occurrences(singular_object, habitats_map)
        list_plural_habitats = find_occurrences(plural_object, habitats_map)
        result_singular_habitats = calculate_result(actions_or_times, list_singular_habitats)
        result_plural_habitats = calculate_result(actions_or_times, list_plural_habitats)
        result = max(result_plural_tools, result_singular_tools, result_plural_habitats, result_singular_habitats)

        if result == 0:
            c = count_raw_occurences(singular_object, question)
            last_rule = "Raw occurrences"
            result = min(c, 1)

        if result == 0:
            last_rule = "Nothing found"

        final_answer = str(result) if result else None
        if not final_answer:
            last_rule = "Nothing found"

        print(f"Last rule = {last_rule}", file=outstream)
        print(f"Final = {final_answer}", file=outstream)
        print(f"Truth = {question.answer}\n", file=outstream)

        details_str = f"Last rule = {last_rule}"
        if (final_answer != question.answer and question.answer != "N/A") \
                or (question.answer == "N/A" and final_answer is not None):
            if more_info.get("dump_logs_for_bad_answers", False):
                print(outstream.getvalue())

        more_info_for_answer = {"source": QuestionAnswererCountingUses.DESCRIPTION,
                                "details_for_excel": details_str}

        return PredictedAnswer(final_answer, raw_question=question.question, confidence=None,
                               more_info=more_info_for_answer)

    def get_object_from_question(self, question: QuestionAnswerRecipe) -> str:
        q = question.question.lower().replace("?", "").replace("how many", "").replace("are used", "").strip()
        return q
