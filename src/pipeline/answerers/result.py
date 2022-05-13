import nltk
from typing import Dict, Any, List, Tuple

from src.annotated_recipe import AnnotatedSentence
from src.pipeline.interface_question_answering import QuestionAnsweringBase, QuestionAnswerRecipe, PredictedAnswer
from src.pipeline.question_category import QuestionCategory


class QuestionAnswererResultV1(QuestionAnsweringBase):
    DESCRIPTION = "QuestionAnswerer extent extractor"

    def answer_a_question(self, question: QuestionAnswerRecipe, question_category: QuestionCategory,
                          more_info: Dict[str, Any] = {}) -> PredictedAnswer:
        """
        Answers the question "to what extent", belonging to class Result. Assumes the answer sought
        consist of a verb, next the word "till" or "until" followed by some phrase ended with
        a punctuation mark.
        :param question: question to be answered
        :param question_category: assumed to be Result, not checked
        :param more_info: ignored
        :return: answer about the extent
        """
        more_info_for_answer = {"source": QuestionAnswererResultV1.DESCRIPTION}

        try:
            question_keywords = self.extract_keywords_from_question(question.question)
            question_verb = question_keywords[0]

            candidate_sentences = self.find_sentences_with_until(question.recipe.annotated_recipe.annotated_sentences)
            target_sentence, max_similarity = self.most_similar_to_question(candidate_sentences, question_keywords)
        except:
            return PredictedAnswer(None, raw_question=question.question, confidence=None,
                                   more_info=more_info_for_answer)

        answer = self.extract_answer_from_sentence(target_sentence, question_verb)

        return PredictedAnswer(answer, raw_question=question.question,
                               confidence=None, more_info=more_info_for_answer)

    def extract_keywords_from_question(self, question: str) -> List[str]:
        """
        Extract verb and nouns assuming that the question has the form "... do you <verb> ... <noun> ..."
        :param question: the question string
        :return: the verb and the nouns as a single list
        """
        keywords = []

        question_words = nltk.tokenize.word_tokenize(question)

        verb_index = question_words.index("you") + 1
        verb = question_words[verb_index].lower()

        keywords.append(verb)

        span_with_nouns = question_words[verb_index:]

        for word, pos in nltk.pos_tag(span_with_nouns):
            if pos == "NN":
                keywords.append(nltk.stem.WordNetLemmatizer().lemmatize(word, pos='v').lower())

        return keywords

    def find_sentences_with_until(self, sentences: List[AnnotatedSentence]) -> List[AnnotatedSentence]:
        """
        Filters sentences with "until" token and its synonyms
        :param sentences: the sentences to choose from
        :return: list of sentences
        """
        sentences_found = []

        for sentence in sentences:
            if "ingredients" in sentence.sentence_id:
                continue

            for token in sentence.annotated_tokens:
                if "B-Result" in token.semantic_roles:
                    sentences_found.append(sentence)

                    break

        return sentences_found

    def most_similar_to_question(self, sentences: List[AnnotatedSentence],
                                 question_keywords: List[str]) -> Tuple[AnnotatedSentence, int]:
        """
        From a list of sentences selects the most similar to the question
        :param sentences: the sentences to select from
        :param question_keywords: representative words from the question
        :return: the most similar sentence
        """
        max_similarity = -1
        max_index = None

        for i, sentence in enumerate(sentences):
            sentence_words = {token.normalized_token for token in sentence.annotated_tokens}
            common_count = len(sentence_words.intersection(question_keywords))

            if common_count > max_similarity:
                max_similarity = common_count
                max_index = i

        if max_similarity < 1:
            raise ValueError("No sentence happened to be similar to the question.")

        return sentences[max_index], max_similarity

    def extract_answer_from_sentence(self, sentence: AnnotatedSentence, question_verb: str) -> str:
        """
        Extracts a span which forms an answer
        :param sentence: the sentence to extract from
        :param question_verb: The main verb from the question. Expected to occur near the answer sought.
        :return: the answer
        """
        answer = ""

        span_start = None
        question_verb_occurred = False
        tokens = sentence.annotated_tokens

        for i, token in enumerate(tokens):
            if question_verb_occurred:
                if "B-Result" in token.semantic_roles:
                    answer += token.raw_token
                    span_start = i

                    break

            elif token.normalized_token == question_verb:
                question_verb_occurred = True

        if span_start is None:
            return None

        i = span_start + 1

        while i < len(tokens) and "I-Result" in tokens[i].semantic_roles:
            answer += " " + tokens[i].raw_token

            i += 1

        return answer
