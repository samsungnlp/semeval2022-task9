from typing import Dict, Any, List, Optional

import nltk
from conllu.models import TokenList

from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory


class QuestionAnswerer1(QuestionAnsweringBase):
    """
    For the question class 1 ("What should be" + participle) we look for the verb in the participle in the question.
    Then we convert it into infinitive form and look up the verb (EVENT, but also B-V and I-V) in the recipe
    and return the PATIENT (related to the verb) with "the"
    """
    DESCRIPTION = "QuestionAnswerer: What should be"

    @staticmethod
    def _get_patients_from_annotated_recipe(sentences, lemma):
        """
        We want to keep this method for a while, despite of being useless; maybe we will return to it at some point
        :param sentences: annotated recipe sentences
        :param lemma: the lemmatized word looked for
        :return: the list of patients belonging to the sentence with the lemma
        """
        answer = []
        for sentence in sentences:
            if lemma not in [token.normalized_token for token in sentence.annotated_tokens]:
                continue
            for token in sentence.annotated_tokens:
                if token.semantic_roles[1] in ['B-Theme', 'I-Theme', 'B-Patient', 'I-Patient']:
                    answer.append(token.normalized_token)
        return answer

    @staticmethod
    def _get_drop_value_from_parsed_recipe(pars: Dict[str, List[TokenList]], lemma: str) -> List[str]:
        """
        Takes a parsed recipe and looks for tokens equal to the lemma
        Parse the token and return drop values from the 'deprel' field
        Example: for deprel string 'Drop=flour.4.1.3:mixture.4.1.6|Tool=whisk.3.1.5|Habitat=bowl.1.1.11'
                 return ['flour', 'mixture']
        :param pars: parsed recipe in the CoNLL-U format
        :param lemma: the lemmatized word looked for
        :return: list of drops taken from the corresponding token(s)
        """
        result = []
        # Iterate recipe steps
        for key in pars:
            if not key.startswith('step'):
                continue
            step = pars[key]
            # Iterate the step. The step variable is of type List[TokenList], hence the nested loop
            for token_list in step:
                for token in token_list:
                    if not token['lemma'] == lemma:
                        continue
                    deprel = token.get('deprel')
                    assert deprel

                    # Parse out relevant words from deprel
                    # deprel example: 'Drop=flour.4.1.3:mixture.4.1.6|Tool=whisk.3.1.5|Habitat=bowl.1.1.11'
                    deprel_split = deprel.split('|')
                    deprel_drops = [field for field in deprel_split if field.startswith('Drop')]
                    if deprel_drops:
                        assert len(deprel_drops) == 1
                        drops = deprel_drops[0].split('=')[1].split(':')  # e.g. ['flour.4.1.3', 'mixture.4.1.6']
                        drop_words = [drop.split('.')[0] for drop in drops]  # e.g. ['flour', 'mixture']
                        result += drop_words
        return result

    @staticmethod
    def _format(answer: List[str]) -> Optional[str]:
        """
        Take the answer as a list of words, possibly empty,
        and format according to the task requirements
        :param answer: list of answer words
        :return: answer formatted as a string or None if the input is empty
        """

        if not answer:
            return None

        # Deduplicate answer preserving the order
        answer = list(dict.fromkeys(answer))

        # Add commas where needed
        if len(answer) > 2:
            answer = [word + ',' for word in answer[:-2]] + answer[-2:]

        # Add 'and' if needed
        if len(answer) > 1 and 'and' not in answer:
            answer = answer[:-1] + ['and'] + answer[-1:]

        # Add 'the' at the beginning
        answer = ['the'] + answer

        # Concatenate
        answer = ' '.join(answer)

        # Replace underscores with spaces
        answer = answer.replace('_', ' ')

        return answer


    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        :param question: question to be answered
        :param question_category: assumed to be 1
        :param more_info: ignored
        :return: predicted answer
        """
        more_info_for_answer = {"source": self.DESCRIPTION}

        # We have checked that all questions have the "what should be pp *" form
        q_words = nltk.tokenize.word_tokenize(question.question)
        q_v_participle = q_words[3]

        q_v_lemma = nltk.stem.WordNetLemmatizer().lemmatize(q_v_participle, pos='v')

        answer_by_patient = self._get_patients_from_annotated_recipe(
            question.recipe.annotated_recipe.annotated_sentences, q_v_lemma)

        answer_by_drop = self._get_drop_value_from_parsed_recipe(question.recipe.new_pars, q_v_lemma)

        # answer = answer_by_patient + answer_by_drop  # This gives F1 0.73, EM 0.43; neither semantic role helps
        answer = answer_by_drop

        answer = self._format(answer)

        return PredictedAnswer(answer, raw_question=question.question,
                               confidence=None, more_info=more_info_for_answer)
