import io
from collections import Counter
from typing import Dict, Any, List

from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.pipeline.verb_object_habitat import VerbPatientHabitat
from src.putty_lemmatizer import PuttyLemmatizer


class QuestionAnswererLocationCrl(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: Where should you?"

    def __init__(self):
        self.lemmatizer = PuttyLemmatizer()
        self.outstream = io.StringIO()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: ignored
        :param more_info: "dump_logs_for_bad_answers" : True if you want the diagnostic logs
        :return:
        """
        more_info_for_answer = {"source": QuestionAnswererLocationCrl.DESCRIPTION}
        self.outstream = io.StringIO()

        print(f"Q = {question.question}", file=self.outstream)
        as_tokens = question.question.replace("?", "").replace(",", " , ").split(" ")
        as_tokens = [x for x in as_tokens if x not in {"a", "the", "an", ""}]
        verb = self.lemmatizer.lemmatize_verb(as_tokens[3])
        print(f"V = {verb}", file=self.outstream)

        objects = self.extract_objects(as_tokens[4:])
        print(f"Objects = {objects}", file=self.outstream)

        events = VerbPatientHabitat.build_list_of_events(question)
        print(f"All Events = {events}", file=self.outstream)
        rule_applied = "exact_match_to_any"

        last_match = self.exact_match_to_any_patients(events, objects, verb)
        print(f"Matching events = {last_match}", file=self.outstream)
        with_nonempty_habitats = [e for e in last_match if e.habitats]
        candidates = [e.habitats[0] for e in with_nonempty_habitats]

        if not candidates:
            print("Trying soft object_search", file=self.outstream)
            last_match = self.soft_match_to_any_patients(verb, objects, events)
            print(f"Soft Matching = {last_match}", file=self.outstream)
            candidates = [e.habitats[0] for e in last_match]
            rule_applied = "soft_match_to_any"

        print(f"Candidates = {candidates}", file=self.outstream)

        final_answer = self.get_final_from_candidates(candidates)
        if not final_answer:
            rule_applied = "Nothing found"

        print(f"final ret = {final_answer}", file=self.outstream)
        print(f"truth = {question.answer}\n\n", file=self.outstream)
        details_str = f"Rule = {rule_applied}"

        if (final_answer != question.answer and question.answer != "N/A") \
                or (question.answer == "N/A" and final_answer is not None):

            details_str += f"\nMatches = {last_match}"
            if more_info.get("dump_logs_for_bad_answers", False):
                print(self.outstream.getvalue())

        more_info_for_answer["details_for_excel"] = details_str

        return PredictedAnswer(final_answer, raw_question=question.question, more_info=more_info_for_answer)

    def soft_match_to_any_patients(self, verb: str, objects: List[str], events: List[VerbPatientHabitat]) \
            -> List[VerbPatientHabitat]:

        def count_soft_matched(e: VerbPatientHabitat, objects: List[str]) -> int:
            return sum([e.is_soft_match_to_any_patients(o) or
                        e.is_soft_match_to_any_patients(self.lemmatizer.lemmatize_noun(o))
                        for o in objects])

        shorter_objects = []
        for object in objects:
            shorter_objects += object.split("_")

        with_verb_and_habitat = [e for e in events if e.verb == verb and e.habitats]
        with_soft_matching = [(e, count_soft_matched(e, shorter_objects)) for e in with_verb_and_habitat]
        with_soft_matching.sort(reverse=True, key=lambda x: x[1])
        return [e for e, c in with_soft_matching if c >= 1]

    def exact_match_to_any_patients(self, events: List[VerbPatientHabitat], objects: List[str], verb: str) \
            -> List[VerbPatientHabitat]:

        def count_matched(e: VerbPatientHabitat, objects: List[str]) -> int:
            return sum([e.is_exact_match_to_any_patients(o) or
                        e.is_exact_match_to_any_patients(self.lemmatizer.lemmatize_noun(o))
                        for o in objects])

        matches = [(e, count_matched(e, objects)) for e in events if e.verb == verb]
        matches.sort(reverse=True, key=lambda x: x[1])
        return [e for e, c in matches if c >= 1]

    def extract_objects(self, tail) -> List[str]:
        ret = []
        current = []
        breaks = ["and", "with", ","]
        for token in tail:
            if token in breaks:
                if current:
                    ret.append("_".join(current))
                current = []
            else:
                current.append(token)
        if current:
            ret.append("_".join(current))
        return ret

    def get_final_from_candidates(self, candidates: List[str]) -> str:
        if not candidates:
            return None
        ret = []
        for c in candidates:
            as_array = c.split("_")
            singular = self.lemmatizer.lemmatize_noun(as_array[-1])
            as_array[-1] = singular
            ret.append(" ".join(as_array))

        if len(ret) >= 2:
            print(f"Non-unique answer to location CRL = {ret}", file=self.outstream)
            c = Counter()
            for item in ret:
                c[item] += 1
            ret = [x[0] for x in c.most_common()]

        return ret[0]
