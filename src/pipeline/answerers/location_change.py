import io
from typing import Dict, Any, List

from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory
from src.pipeline.verb_object_habitat import VerbPatientHabitat
from src.putty_lemmatizer import PuttyLemmatizer


class QuestionAnswererLocationChange(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer: Where was X before Y?"

    def __init__(self):
        self.lemmatizer = PuttyLemmatizer()
        self.outstream = io.StringIO()

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: ignored
        :param more_info: "dump_logs_for_bad_answers" : True if you want the diagnostic logs
        :return: answer
        """
        more_info_for_answer = {"source": QuestionAnswererLocationChange.DESCRIPTION}

        answer2, diagnostics = self.try_answer_v3_96(question, more_info)
        more_info_for_answer["details_for_excel"] = diagnostics

        final_answer = answer2
        return PredictedAnswer(final_answer, raw_question=question.question, more_info=more_info_for_answer)

    def try_answer_v3_96(self, question, more_info) -> str:
        """
        1. searches for events in recipe (verb + object  + location)
        """
        self.outstream = io.StringIO()

        print(f"RID = {question.recipe.id}", file=self.outstream)
        print(f"Q = {question.question}", file=self.outstream)
        an_object = self.get_subject_from_question(question.question)
        print(f"S = {an_object}", file=self.outstream)
        verb = QuestionAnswererLocationChange.get_reference_verb_from_question(question.question)
        lemmatized_verb = self.lemmatizer.lemmatize_verb(verb)
        print(f"V = {lemmatized_verb}", file=self.outstream)
        context = QuestionAnswererLocationChange.get_question_context(question.question)
        vphs = VerbPatientHabitat.build_list_of_events_for_c17(question)
        print(f"All_events = {vphs}", file=self.outstream)
        aliases = self.search_for_aliases(an_object, question)
        print(f"Aliases = {aliases}", file=self.outstream)

        context_event = self.find_context_events(vphs, lemmatized_verb, an_object, context)
        if len(context_event) >= 2:
            print(f"Nonunique context events = {context_event}", file=self.outstream)
        context_event = context_event[0] if len(context_event) == 1 else None
        print(f"Context event = {context_event}", file=self.outstream)

        exact_subject_matching = []
        rule_applied = "ContextMatch"
        soft_matching = []
        last_matches = []
        candidates = []

        if not candidates:
            rule_applied = "ExactMatch Prev"
            print("Trying exact match", file=self.outstream)
            exact_subject_matching = self.find_exact_matches(an_object, vphs)
            last_matches = exact_subject_matching
            print(f"Matching = {exact_subject_matching}", file=self.outstream)
            candidates = self.__extract_prev_habitats_from_matches(lemmatized_verb, context_event,
                                                                   exact_subject_matching)

        for alias in aliases:
            if not candidates:
                rule_applied = f"AliasesMatch \"{alias}\" Prev"
                print(f"Trying alias match vs {alias}", file=self.outstream)
                alias_matching = self.find_exact_matches(alias, vphs)
                last_matches = alias_matching
                print(f"Matching to {alias}= {alias_matching}", file=self.outstream)
                candidates = self.__extract_prev_habitats_from_matches(lemmatized_verb, context_event, alias_matching)

        if not candidates:
            rule_applied = "SoftMatch Prev"
            print("Trying soft subject_search", file=self.outstream)
            soft_matching = self.find_soft_matches(an_object, vphs)
            last_matches = soft_matching
            print(f"Soft Matching = {soft_matching}", file=self.outstream)
            candidates = self.__extract_prev_habitats_from_matches(lemmatized_verb, context_event, soft_matching)

        if not candidates:
            rule_applied = "ContextFoundPatients Prev"
            print("Trying patients from context event match", file=self.outstream)
            context_matchings = self.__find_context_matches(vphs, context_event)
            last_matches = context_matchings
            print(f"Matching = {context_matchings}", file=self.outstream)
            candidates = self.__extract_prev_habitats_from_matches(lemmatized_verb, context_event, context_matchings)

        if not candidates:
            rule_applied = "ExactMatch Current"
            print("Trying current habitat", file=self.outstream)
            last_matches = exact_subject_matching
            candidates = self.__extract_current_habitats_from_matches(lemmatized_verb, exact_subject_matching)

        if not candidates:
            rule_applied = "SoftMatch Current"
            print("Trying current soft-matched habitat", file=self.outstream)
            last_matches = soft_matching
            candidates = self.__extract_current_habitats_from_matches(lemmatized_verb, soft_matching)

        if not candidates:
            rule_applied = "Nothing found"

        print(f"Candidates = {candidates}", file=self.outstream)
        final_answer = self.get_final_from_candidates(candidates)

        print(f"final ret = {final_answer}", file=self.outstream)
        print(f"truth = {question.answer}\n\n", file=self.outstream)
        diagnostics = f"Rule applied = {rule_applied}\n"

        if final_answer != question.answer and question.answer != "N/A":
            if more_info.get("dump_logs_for_bad_answers", False):
                print(self.outstream.getvalue())
            diagnostics += f"Candidates = {candidates}\n"
            diagnostics += f"Matches = {last_matches}"
        return final_answer, diagnostics

    def find_exact_matches(self, an_object: str, all_events: List[VerbPatientHabitat]) -> List[VerbPatientHabitat]:
        return [verb_patient for verb_patient in all_events if verb_patient.is_exact_match_to_any_patients(an_object)]

    def search_for_aliases(self, an_object: str, ar: QuestionAnswerRecipe) -> List[str]:
        objects_as_list = an_object.lower().split("_")
        l = len(objects_as_list)
        ret = []

        for sentence in ar.recipe.annotated_recipe.annotated_sentences:
            for token in sentence.annotated_tokens:
                i = token.id - 1
                tokens = sentence.annotated_tokens[i:i + l] \
                    if i + l < len(sentence.annotated_tokens) else sentence.annotated_tokens[i:]
                words = [t.raw_token.lower() for t in tokens]

                if words == objects_as_list:
                    aliases = [t.relation2.split(".")[0] for t in tokens if t.relation2]
                    aliases = [a for a in aliases if a != an_object]
                    ret.extend(aliases)

        return list(set(ret))

    def find_soft_matches(self, subject, vphs) -> List[VerbPatientHabitat]:
        shorter_subject = [x for x in subject.split("_") if x]
        soft_matching = [verb_patient for verb_patient in vphs
                         if any(verb_patient.is_soft_match_to_any_patients(subj) for subj in shorter_subject)]
        return soft_matching

    def __find_context_matches(self, all_events: List[VerbPatientHabitat], context_event: VerbPatientHabitat) \
            -> List[VerbPatientHabitat]:
        if not context_event:
            return []

        prev_events = [e for e in all_events if
                       e.sentence_id < context_event.sentence_id
                       or (e.sentence_id == context_event.sentence_id and e.token_id < context_event.token_id)]

        matching_events = [e for e in prev_events if len(context_event.get_common_patients(e)) >= 1]
        return matching_events + [context_event]

    def find_context_events(self, all_events: List[VerbPatientHabitat], verb: str, _: str, context: List[str]) \
            -> List[VerbPatientHabitat]:
        ret = []
        for event in all_events:
            if verb and verb != event.verb:
                continue
            found_words = [word in event.all_related_words for word in context]
            if context and found_words.count(True) == len(context) and event.patients:
                ret.append(event)
        return ret

    def get_final_from_candidates(self, candidates):
        if not candidates:
            return None
        ret = []
        for c in candidates:
            as_array = c.split(".")[0].split(" ")
            singular = self.lemmatizer.lemmatize_noun(as_array[-1])
            as_array[-1] = singular
            ret.append(" ".join(as_array))

        if len(ret) != 1:
            # TODO handle nonunique answers
            print(f"Non-unique answer to LocationChange = {ret}", file=self.outstream)
        return ret[-1]

    def __extract_prev_habitats_from_matches(self, lemmatized_verb: str, context_event: VerbPatientHabitat,
                                             matches: List[VerbPatientHabitat]) -> List[str]:

        context_event_indices = [i for i, e in enumerate(matches) if e.sentence_id == context_event.sentence_id
                                 and e.token_id == context_event.token_id] if context_event else []

        indices_of_verbs = context_event_indices if context_event_indices else \
            [i for i, verb_patient in enumerate(matches) if verb_patient.verb == lemmatized_verb]

        returns = self.find_previous_event_locations(indices_of_verbs, matches)
        not_nones = [x for x in returns if x]
        return not_nones

    def __extract_current_habitats_from_matches(self, lemmatized_verb: str, matches: List[VerbPatientHabitat]) \
            -> List[str]:
        indices_of_verbs = [i for i, verb_patient in enumerate(matches) if
                            verb_patient.verb == lemmatized_verb]
        returns = self.find_current_event_locations(indices_of_verbs, matches)
        not_nones = [x for x in returns if x]
        return not_nones

    def find_previous_event_locations(self, indices_of_verbs: List[int], only_matching: List[VerbPatientHabitat]) \
            -> List[str]:
        returns = []
        for index in indices_of_verbs:
            current_habitat = only_matching[index].habitats[0] if only_matching[index].habitats else None
            current_coref = current_habitat[current_habitat.find("."):] if current_habitat else None

            for j in range(index - 1, -1, -1):
                reference_habitats = only_matching[j].habitats

                if not reference_habitats:
                    continue
                reference_coref = reference_habitats[0][reference_habitats[0].find("."):] \
                    if reference_habitats[0] else None
                if not current_habitat or (
                        current_habitat != reference_habitats[0] and reference_coref != current_coref):
                    returns.append(reference_habitats[0].replace("_", " "))
                    break
                returns.append(None)
        return returns

    def find_current_event_locations(self, indices_of_verbs: List[int], only_matching: List[VerbPatientHabitat]) \
            -> List[str]:
        returns = []
        for index in indices_of_verbs:
            for j in range(index, -1, -1):
                if only_matching[j].habitats:
                    returns.append(only_matching[j].habitats[0].replace("_", " "))
                    break
                returns.append(None)
        return returns

    def get_subject_from_question(self, question: str) -> str:
        start_start = question.find("Where was")
        start_end = start_start + len("Where was") if start_start >= 0 else -1

        end = question.find("before")
        if start_end == -1 or end == -1:
            raise ValueError(f"Cannot find subject in question {question}")

        subject = question[start_end:end].strip()
        if subject.find("the ") == 0:
            subject = subject[4:]
        if subject.find("an ") == 0:
            subject = subject[3:]
        if subject.find("a ") == 0:
            subject = subject[2:]

        as_array = subject.split(" ")
        as_array[-1] = self.lemmatizer.lemmatize_noun(as_array[-1])
        subject = "_".join(as_array)
        return subject

    @staticmethod
    def get_reference_verb_from_question(question: str) -> str:
        start_b_i_was = question.find("before it was")
        end = start_b_i_was + len("before it was") if start_b_i_was >= 0 else -1

        if end >= 0:
            verb = question[end:].split()[0].strip()
            verb = verb.replace("?", "")
            return verb

        raise ValueError(f"Cannot find verb in question {question}")

    @staticmethod
    def get_question_context(question: str) -> List[str]:
        start_b_i_was = question.find("before it was ")
        end = start_b_i_was + len("before it was ") if start_b_i_was >= 0 else -1

        suffix = question[end:].replace("?", "").split()
        return suffix[1:]
