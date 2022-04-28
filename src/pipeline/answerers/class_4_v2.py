import io
from typing import Dict, Any, List

import nltk

from src.pipeline.answerers.class_4 import _return_answer
from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.pipeline.verb_object_habitat import VerbPatientHabitat
from src.putty_lemmatizer import PuttyLemmatizer


class C4Event:

    def __init__(self, verb: str, objects: List[str], all_words: List[str]) -> None:
        self.verb = verb
        self.objects = objects
        self.all_words = all_words

    def __repr__(self):
        return f"{self.verb} {self.objects} ({self.all_words})"

    def __str__(self):
        return repr(self)

    def is_exact_match(self, e1: VerbPatientHabitat) -> bool:
        return self.is_verb_match(e1) and self.is_exact_object_match(e1)

    def is_soft_match(self, e1: VerbPatientHabitat) -> bool:
        return self.is_verb_match(e1) and self.is_soft_object_match(e1)

    def is_verb_match(self, e1: VerbPatientHabitat) -> bool:
        return e1.verb == self.verb

    def is_exact_object_match(self, e1: VerbPatientHabitat) -> bool:
        ref_objects = set(e1.patients + e1.habitats)
        return len(set(self.objects).intersection(ref_objects)) != 0

    def is_soft_object_match(self, e1: VerbPatientHabitat) -> bool:
        ref_objects = e1.patients + e1.habitats
        return any(my_object in ref_object for my_object in self.objects for ref_object in ref_objects)

    def is_object_as_verb_match(self, e1: VerbPatientHabitat) -> bool:
        return any(my_object == e1.verb for my_object in self.objects)

    def full_sentence_match(self, e1: VerbPatientHabitat) -> bool:
        if not self.is_verb_match(e1):
            return False

        restricted_tokens = {"the", "a", "an"}
        my_words = self.all_words[1:]  # skip first verb
        my_words = set(x for x in my_words if x not in restricted_tokens)
        event_words = set(x for x in e1.all_related_words if x not in restricted_tokens)

        intersection = my_words.intersection(event_words)
        return len(intersection) == len(my_words) or (len(intersection) == len(my_words) - 1 >= 2)


class QuestionAnswerer4EventBased(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: A,B Which comes first? (Event Based)"

    def __init__(self):
        self.lemmatizer = PuttyLemmatizer()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: ignored
        :param more_info: "dump_logs_for_bad_answers" : True if you want the diagnostic logs
        :return:
        """
        more_info_for_answer = {"source": QuestionAnswerer4EventBased.DESCRIPTION}
        outstream = io.StringIO()

        print(f"Q = {question.question}", file=outstream)
        segments = self.split_question_into_events(question.question)

        for i, s in enumerate(segments):
            events = self.extract_events_from_segment(s)
            print(f"Events{i} = {events}", file=outstream)
        if len(segments) != 2:
            print(f"\nNum segments != 2   = {segments}!!!\n", file=outstream)

        first_event: C4Event = self.extract_events_from_segment(segments[0])
        last_event: C4Event = self.extract_events_from_segment(segments[-1])

        all_events = VerbPatientHabitat.build_list_of_events(question)
        print(f"all events = {all_events}", file=outstream)

        final_answer = ""
        last_match_1 = []
        last_match_2 = []

        if not final_answer:
            rule_applied = "Full sentence match"
            last_match_1 = last_match_1 if last_match_1 else self.full_sentence_match(all_events, first_event)
            print(f"{rule_applied}1 = {last_match_1}", file=outstream)
            last_match_2 = last_match_2 if last_match_2 else self.full_sentence_match(all_events, last_event)
            print(f"{rule_applied}2 = {last_match_2}", file=outstream)
            final_answer = self.compare_by_positions(last_match_1, last_match_2, outstream)

        if not final_answer:
            rule_applied = "Exact matched"
            last_match_1 = last_match_1 if last_match_1 else self.exact_match(all_events, first_event)
            print(f"{rule_applied}1 = {last_match_1}", file=outstream)
            last_match_2 = last_match_2 if last_match_2 else self.exact_match(all_events, last_event)
            print(f"{rule_applied}2 = {last_match_2}", file=outstream)
            final_answer = self.compare_by_positions(last_match_1, last_match_2, outstream)

        if not final_answer:
            rule_applied = "Soft match"
            last_match_1 = last_match_1 if last_match_1 \
                else QuestionAnswerer4EventBased.soft_match(all_events, first_event)
            print(f"{rule_applied}1 = {last_match_1}", file=outstream)
            last_match_2 = last_match_2 if last_match_2 \
                else QuestionAnswerer4EventBased.soft_match(all_events, last_event)
            print(f"{rule_applied}2 = {last_match_2}", file=outstream)
            final_answer = self.compare_by_positions(last_match_1, last_match_2, outstream)

        if not final_answer:
            rule_applied = "Only Verb Match"
            last_match_1 = last_match_1 if last_match_1 \
                else QuestionAnswerer4EventBased.verb_match(all_events, first_event)
            last_match_2 = last_match_2 if last_match_2 \
                else QuestionAnswerer4EventBased.verb_match(all_events, last_event)
            print(f"Only Verb matched1 = {last_match_1}", file=outstream)
            print(f"Only Verb matched2 = {last_match_2}", file=outstream)
            final_answer = self.compare_by_positions(last_match_1, last_match_2, outstream)

        if not final_answer:
            rule_applied = "Only Obj Match"
            last_match_1 = last_match_1 if last_match_1 else self.soft_object_match(all_events, first_event)
            last_match_2 = last_match_2 if last_match_2 else self.soft_object_match(all_events, last_event)
            print(f"Only Obj matched1 = {last_match_1}", file=outstream)
            print(f"Only Obj matched2 = {last_match_2}", file=outstream)
            final_answer = self.compare_by_positions(last_match_1, last_match_2, outstream)

        if not final_answer:
            rule_applied = "Object to Verb Match"
            # pathological case: match objects vs verbs
            last_match_1 = last_match_1 if last_match_1 else self.object_to_verb_match(all_events, first_event)
            last_match_2 = last_match_2 if last_match_2 else self.object_to_verb_match(all_events, last_event)
            print(f"Obj-Verb matched1 = {last_match_1}", file=outstream)
            print(f"Obj-Verb matched2 = {last_match_2}", file=outstream)
            final_answer = self.compare_by_positions(last_match_1, last_match_2, outstream)

        if not final_answer:
            rule_applied = "Edit distance"
            print(f"Compare with edit distance", file=outstream)
            final_answer = _return_answer(question, threshold=0.35)

        if not final_answer:
            rule_applied = "Stupid Cases Match"
            print(f"Handling stupid cases:", file=outstream)
            final_answer = self.hanldle_stupid_cases(first_event, last_event)

        if not final_answer:
            rule_applied = "Nothing found"

        print(f"final answer = {final_answer}", file=outstream)
        print(f"truth = {question.answer}\n\n", file=outstream)

        if (final_answer != question.answer and question.answer != "N/A") \
                or (question.answer == "N/A" and final_answer is not None):
            if more_info.get("dump_logs_for_bad_answers", False):
                print(outstream.getvalue())

        msg = f"Rule_applied = {rule_applied}"
        if final_answer != question.answer:
            msg += f"\nLast match 1 = {last_match_1}\n" \
                   f"Last match 2 = {last_match_2}"
        more_info_for_answer["details_for_excel"] = msg
        return PredictedAnswer(final_answer, more_info=more_info_for_answer)

    @staticmethod
    def verb_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) -> List[VerbPatientHabitat]:
        return [event for event in all_events if reference_event.is_verb_match(event)]

    @staticmethod
    def exact_object_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) -> List[VerbPatientHabitat]:
        return [event for event in all_events if reference_event.is_exact_object_match(event)]

    @staticmethod
    def soft_object_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) -> List[VerbPatientHabitat]:
        return [event for event in all_events if reference_event.is_soft_object_match(event)]

    @staticmethod
    def soft_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) -> List[VerbPatientHabitat]:
        return [event for event in all_events if reference_event.is_soft_match(event)]

    @staticmethod
    def exact_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) -> List[VerbPatientHabitat]:
        return [event for event in all_events if reference_event.is_exact_match(event)]

    @staticmethod
    def full_sentence_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) -> List[VerbPatientHabitat]:
        ret = [event for event in all_events if reference_event.full_sentence_match(event)]
        # if len(ret) >= 2:
        #     print(f"Non unique events found for {reference_event}:  {ret}")
        return ret

    @staticmethod
    def object_to_verb_match(all_events: List[VerbPatientHabitat], reference_event: C4Event) \
            -> List[VerbPatientHabitat]:
        """
        :param all_events:
        :param reference_event:
        :return: list of events from all event whose verb is and object (!!!) in reference events (pathological case)
        """
        return [event for event in all_events if reference_event.is_object_as_verb_match(event)]

    @staticmethod
    def hanldle_stupid_cases(event1: C4Event, event2: C4Event) -> str:
        """
        "X and X, which comes first?"
        and other pathological cases
        """
        if event1.all_words == event2.all_words:
            return "the first event"
        return None

    def compare_by_positions(self, matched_events1: List[VerbPatientHabitat],
                             matched_events2: List[VerbPatientHabitat],
                             outstream=None) -> str:
        final_answer = self.compare_using_sentence_position(matched_events1, matched_events2, None)
        if not final_answer:
            final_answer = self.compare_using_token_position(matched_events1, matched_events2, None)
        return final_answer

    def compare_using_sentence_position(self, matched_events1, matched_events2, outstream=None) -> str:
        final_answer = None
        s1 = [e.sentence_id for e in matched_events1]
        s2 = [e.sentence_id for e in matched_events2]
        avg_s1 = self.avg(s1)
        avg_s2 = self.avg(s2)
        if outstream:
            print(f"avg_sentence pos = {avg_s1} vs {avg_s2}", file=outstream)

        if any([avg_s1 is None, avg_s2 is None]):
            return None
        if avg_s1 < avg_s2:
            final_answer = "the first event"
        elif avg_s1 > avg_s2:
            final_answer = "the second event"
        return final_answer

    def compare_using_token_position(self, matched_events1, matched_events2, outstream=None):
        avg_t1 = self.avg([e.token_id for e in matched_events1])
        avg_t2 = self.avg([e.token_id for e in matched_events2])
        if outstream:
            print(f"avg_token pos = {avg_t1} vs {avg_t2}", file=outstream)
        if any([avg_t1 is None, avg_t2 is None]):
            return None
        if avg_t1 < avg_t2:
            return "the first event"
        elif avg_t1 > avg_t2:
            return "the second event"
        return None

    def avg(self, t1):
        return sum(t1) / len(t1) if t1 else None

    def split_question_into_events(self, question: str) -> List[List[str]]:
        q = question.replace("which comes first?", "").replace(",", " ").lower()
        as_array = [x for x in q.split(" ") if x]
        lemmatized = nltk.pos_tag(as_array)

        ret: List[List[str]] = []
        current: List[str] = []
        for i, token in enumerate(lemmatized):
            # print(f"{token}   {token[0][-3:]}")

            if i >= 1 and lemmatized[i - 1][0] == "and" \
                    and (token[1] == "VBG" or token[1] == "NN" and token[0][-3:] == "ing"):
                ret.append(current[0:-1])
                current = [token[0]]
            else:
                current.append(token[0])
        if current:
            ret.append(current)

        return ret

    def extract_events_from_segment(self, segment: List[str]) -> C4Event:

        lemmatized_verb = self.lemmatizer.lemmatize_verb(segment[0])

        rets = []
        pos_tagged = nltk.pos_tag(segment)
        prev_modifiers = []
        for token, pos in pos_tagged:
            if pos in {"NN", "NNS", "PRP"}:
                obj = "_".join(prev_modifiers + [self.lemmatizer.lemmatize_noun(token)])
                rets.append(obj)
                prev_modifiers = []
            elif pos in {"VBD", "JJ"}:
                prev_modifiers.append(token)

        return C4Event(lemmatized_verb, rets, segment)
