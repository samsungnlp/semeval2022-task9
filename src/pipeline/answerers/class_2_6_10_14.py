from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from typing import Dict, List, Any
from src.pipeline.answerers.class_2_how import QuestionAnswerer2How
from src.pipeline.answerers.class_6_how import QuestionAnswerer6How
from src.pipeline.answerers.class_10_how import QuestionAnswerer10How
from src.pipeline.answerers.class_14_how import QuestionAnswerer14How


class QuestionAnswerer2_6_10_14(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: How do you?"

    def __init__(self, class2_semantic_roles: List[str], class2_answer_annotations: List[str],
                 class6_semantic_roles: List[str], class6_answer_annotations: List[str],
                 class10_semantic_roles: List[str], class10_answer_annotations: List[str],
                 class14_semantic_roles: List[str], class14_answer_annotations: List[str]):

        self.class2_semantic_roles = class2_semantic_roles
        self.class2_answer_annotations = class2_answer_annotations
        self.class6_semantic_roles = class6_semantic_roles
        self.class6_answer_annotations = class6_answer_annotations
        self.class10_semantic_roles = class10_semantic_roles
        self.class10_answer_annotations = class10_answer_annotations
        self.class14_semantic_roles = class14_semantic_roles
        self.class14_answer_annotations = class14_answer_annotations

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        Answers the question "how do you", belonging to class 2, 6, 10 and 14. Iterate over paragraphs and semantic
        role's columns to find specified labels.
        :param question: question to be answered
        :param question_category: assumed to from 2, 6, 10 or 14
        :param more_info: ignored
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswerer2_6_10_14.DESCRIPTION}
        answers = []
        verb = question.question.split()[3]
        last = question.question.replace("?", "").split()[-1]

        if verb in ["use", "cool"]:
            answer14, more_info_for_answer14 = QuestionAnswerer14How(
                self.class14_semantic_roles, self.class14_answer_annotations).answer_class14_question(question)
            if answer14:
                answers.append(answer14)

        elif verb in ["fry", "stir"] or last in ["minutes", "well", "gently"]:
            answer6, more_info_for_answer6 = QuestionAnswerer6How(
                self.class6_semantic_roles, self.class6_answer_annotations).answer_class6_question(question)
            if answer6:
                answers.append(answer6)

        elif (verb in ["mix", "beat", "stir"] and last == "bowl") or last in ["mixture", "bowl"]:
            answer2, more_info_for_answer2 = QuestionAnswerer2How(
                self.class2_semantic_roles, self.class2_answer_annotations).answer_class2_question(question)
            if answer2:
                answers.append(answer2)

        answer10, more_info_for_answer10 = QuestionAnswerer10How(
            self.class10_semantic_roles, self.class10_answer_annotations).answer_class10_question(question)
        if answer10:
            answers.append(answer10)

        answer6, more_info_for_answer6 = QuestionAnswerer6How(
            self.class6_semantic_roles, self.class6_answer_annotations).answer_class6_question(question)
        if answer6:
            answers.append(answer6)

        answer14, more_info_for_answer14 = QuestionAnswerer14How(
            self.class14_semantic_roles, self.class14_answer_annotations).answer_class14_question(question)
        if answer14:
            answers.append(answer14)

        answer2, more_info_for_answer2 = QuestionAnswerer2How(
            self.class2_semantic_roles, self.class2_answer_annotations).answer_class2_question(question)
        if answer2:
            answers.append(answer2)

        if answers:
            more_info_for_answer["details_for_excel"] = f"answer2: {answer2} || answer6: {answer6} || answer10: {answer10} || answer14: {answer14}"
            return PredictedAnswer(answers[0], raw_question=question.question, confidence=None,
                                   more_info=more_info_for_answer)

        more_info_for_answer["details_for_excel"] = "no match"
        return PredictedAnswer(None, raw_question=question.question, confidence=None, more_info=more_info_for_answer)
