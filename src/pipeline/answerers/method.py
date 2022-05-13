from typing import Dict, List, Any

from src.pipeline.answerers.method_attribute import QuestionAnswererMethodAttribute
from src.pipeline.answerers.method_goal import QuestionAnswererMethodGoal
from src.pipeline.answerers.method_instrument import QuestionAnswererMethodInstrument
from src.pipeline.answerers.method_tool import QuestionAnswererMethodTool
from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory


class QuestionAnswererMethod(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: How do you?"

    def __init__(self, tool_semantic_roles: List[str], tool_answer_annotations: List[str],
                 instrument_semantic_roles: List[str], instrument_answer_annotations: List[str],
                 attribute_semantic_roles: List[str], attribute_answer_annotations: List[str],
                 goal_semantic_roles: List[str], goal_answer_annotations: List[str]):

        self.tool_semantic_roles = tool_semantic_roles
        self.tool_answer_annotations = tool_answer_annotations
        self.instrument_semantic_roles = instrument_semantic_roles
        self.instrument_answer_annotations = instrument_answer_annotations
        self.attribute_semantic_roles = attribute_semantic_roles
        self.attribute_answer_annotations = attribute_answer_annotations
        self.goal_semantic_roles = goal_semantic_roles
        self.goal_answer_annotations = goal_answer_annotations

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        Answers the question "how do you", belonging to class method 
        (original tool, instrument, attribute and goal, cannot tell which exactly).
        Iterate over paragraphs and semantic
        role's columns to find specified labels.
        :param question: question to be answered
        :param question_category: assumed to be method
        :param more_info: ignored
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswererMethod.DESCRIPTION}
        answers = []
        verb = question.question.split()[3]
        last = question.question.replace("?", "").split()[-1]

        if verb in ["use", "cool"]:
            answer_goal, more_info_for_answer_goal = QuestionAnswererMethodGoal(
                self.goal_semantic_roles, self.goal_answer_annotations).answer_method_goal_question(question)
            if answer_goal:
                answers.append(answer_goal)

        elif verb in ["fry", "stir"] or last in ["minutes", "well", "gently"]:
            answer_instrument, more_info_for_answer_instrument = QuestionAnswererMethodInstrument(
                self.instrument_semantic_roles, self.instrument_answer_annotations).answer_class_method_instrument_question(question)
            if answer_instrument:
                answers.append(answer_instrument)

        elif (verb in ["mix", "beat", "stir"] and last == "bowl") or last in ["mixture", "bowl"]:
            answer_tool, more_info_for_answer_tool = QuestionAnswererMethodTool(
                self.tool_semantic_roles, self.tool_answer_annotations).answer_method_tool_question(question)
            if answer_tool:
                answers.append(answer_tool)

        answer_attribute, more_info_for_answer_attribute = QuestionAnswererMethodAttribute(
            self.attribute_semantic_roles, self.attribute_answer_annotations).answer_method_attribute_question(question)
        if answer_attribute:
            answers.append(answer_attribute)

        answer_instrument, more_info_for_answer_instrument = QuestionAnswererMethodInstrument(
            self.instrument_semantic_roles, self.instrument_answer_annotations).answer_class_method_instrument_question(question)
        if answer_instrument:
            answers.append(answer_instrument)

        answer_goal, more_info_for_answer_goal = QuestionAnswererMethodGoal(
            self.goal_semantic_roles, self.goal_answer_annotations).answer_method_goal_question(question)
        if answer_goal:
            answers.append(answer_goal)

        answer_tool, more_info_for_answer_tool = QuestionAnswererMethodTool(
            self.tool_semantic_roles, self.tool_answer_annotations).answer_method_tool_question(question)
        if answer_tool:
            answers.append(answer_tool)

        if answers:
            more_info_for_answer["details_for_excel"] = f"tool: {answer_tool} || instrument: {answer_instrument} " \
                                                        f"|| attribute: {answer_attribute} || goal: {answer_goal}"
            return PredictedAnswer(answers[0], raw_question=question.question, confidence=None,
                                   more_info=more_info_for_answer)

        more_info_for_answer["details_for_excel"] = "no match"
        return PredictedAnswer(None, raw_question=question.question, confidence=None, more_info=more_info_for_answer)
