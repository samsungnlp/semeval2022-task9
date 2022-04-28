from typing import List, Set, Union, Optional


class AnnotatedToken:
    ALLOWED_POS: Set[str] = {"NUM", "PUNCT", "NOUN", "VERB", "ADJ", "ADV", "ADP", "SCONJ", "CCONJ", "AUX"}
    ALLOWED_ROLES = {"EVENT", "EXPLICIT_INGREDIENT", "IMPLICIT_INGREDIENT", "O"}
    SEPARATOR = "\t"

    def __init__(self, id: int, raw_token: str, normalized_token: str, part_of_speech: str):
        self.id: int = id
        self.raw_token: str = raw_token
        self.normalized_token: str = normalized_token
        self.part_of_speech: str = part_of_speech
        self.role_in_recipe: str = None  # col E
        self.where_is_my_verb_explicit: int = None  # col F
        self.where_is_my_verb_implicit: int = None  # col G
        self.relation1: str = None  # colH # verb valentions??
        self.relation2: str = None  # colI #
        self.verb_group: str = None  # colJ
        self.semantic_roles = [None] * 10  # col K -- T
        self.position_in_the_whole_recipe = self.id

    @staticmethod
    def parse_from_line(a_line: str, token_offset: int = 0) -> "AnnotatedToken":
        as_array = a_line.strip().split(AnnotatedToken.SEPARATOR)
        if len(as_array) < 10:
            raise ValueError("Cannot parse line")
        id = int(as_array[0])
        raw_token = as_array[1]
        normalized = as_array[2]
        pos = as_array[3]
        ret = AnnotatedToken(id, raw_token, normalized, pos)
        ret.position_in_the_whole_recipe = token_offset + id - 1

        ret.role_in_recipe = as_array[4] if as_array[4] != "_" else None
        ret.where_is_my_verb_explicit = int(as_array[5]) if as_array[5] != "_" else None
        ret.where_is_my_verb_implicit = int(as_array[6]) if as_array[6] != "_" else None
        ret.relation1 = as_array[7] if as_array[7] != "_" else None
        ret.relation2 = as_array[8] if as_array[8] != "_" else None
        ret.verb_group = as_array[9] if as_array[9] != "_" else None
        ret.semantic_roles = [x if x != "_" else None for x in as_array[10:20]]
        return ret

    def get_entry_from_relation1(self, label: str) -> List[str]:
        as_str = self.relation1 if self.relation1 else ""
        as_array = as_str.split("|")

        ret = []
        for s in as_array:
            if s.find(label) == 0:
                rhs = s.split("=")[1].split(":")
                ret.extend([x.split(".")[0].lower().strip("_") for x in rhs])
        return ret

    def get_whole_entry_from_relation1(self, label: str) -> List[str]:
        as_str = self.relation1 if self.relation1 else ""
        as_array = as_str.split("|")

        ret = []
        for s in as_array:
            if s.find(label) == 0:
                rhs = s.split("=")[1].split(":")
                ret.extend([x.lower().strip("_") for x in rhs])
        return ret

    def is_equal_to_any_verb_id(self, id: int) -> bool:
        return id in [self.where_is_my_verb_explicit, self.where_is_my_verb_implicit]

    def __str__(self):
        return f"{self.raw_token}|{self.id}|{self.relation1}|{self.relation2}"

    def __repr__(self):
        return str(self)


class AnnotatedSentence:
    def __init__(self, list_of_tokens: List[AnnotatedToken], raw_sentence: str):
        self.annotated_tokens: List[AnnotatedToken] = list_of_tokens
        self.raw_sentence: str = raw_sentence
        self.paragraph_id: str = None
        self.sentence_id: str = None
        self.sentence_position_in_paragraph: int = None

    @staticmethod
    def parse_sentence_from_lines(lines: List[str], sentence_offset: int = 0, token_offset: int = 0):

        tokens = []
        for line in lines:
            if line.count("\t") >= 19:
                tokens.append(AnnotatedToken.parse_from_line(a_line=line, token_offset=token_offset))
        sentence_id = _find_value_in_lines(lines, "# sent_id =")
        paragraph_id = sentence_id.rsplit("::", 1)[0]
        raw_text = _find_value_in_lines(lines, "# text =")

        ret = AnnotatedSentence(tokens, raw_text)
        ret.paragraph_id = paragraph_id
        ret.sentence_id = sentence_id
        ret.sentence_position_in_paragraph = sentence_offset
        return ret

    def __str__(self):
        return f"{self.sentence_id}||{self.raw_sentence}||{self.annotated_tokens}"

    def __repr__(self):
        return str(self)


class AnnotatedRecipe:

    def __init__(self, annotated_sentences: List[AnnotatedSentence], raw_recipe: str):
        self.annotated_sentences: List[AnnotatedSentence] = annotated_sentences
        self.raw_recipe: str = raw_recipe
        self.recipe_id: str = None
        self.url: str = None
        self.num_steps: int = None
        self.avg_len_steps: int = None
        self.num_ingredients: int = None
        self.cluster: Union[int, str] = None

    @staticmethod
    def parse_recipe_from_lines(lines: List[str]) -> "AnnotatedRecipe":
        list_of_sentences = AnnotatedRecipe._collect_sentences(lines)
        raw_recipe = "\n".join(x.raw_sentence for x in list_of_sentences)

        ret = AnnotatedRecipe(list_of_sentences, raw_recipe)
        ret.url = _find_value_in_lines(lines, "# metadata:url =")
        ret.num_steps = _find_int_value_in_lines(lines, "# metadata:num_steps =")
        ret.avg_len_steps = _find_int_value_in_lines(lines, "# metadata:avg_len_steps =")
        ret.num_ingredients = _find_int_value_in_lines(lines, "# metadata:num_ingres =")
        value = _find_value_in_lines(lines, "# metadata:cluster =")
        ret.cluster = int(value) if value.isnumeric() else value
        ret.recipe_id = _find_value_in_lines(lines, "# newdoc id =")
        return ret

    @staticmethod
    def _collect_sentences(lines: List[str]) -> List[AnnotatedSentence]:
        new_paragraphs = [line_no for line_no, line in enumerate(lines) if line.find("# sent_id =") == 0]
        sentence_offset = 0
        token_offset = 0
        sentences = []
        for i, line_no in enumerate(new_paragraphs):
            first_line = line_no - 1 if line_no >= 1 else line_no
            end_line = new_paragraphs[i + 1] if (i + 1) < len(new_paragraphs) else len(lines)
            sentence_lines = lines[first_line:end_line]
            a_sentence = AnnotatedSentence.parse_sentence_from_lines(sentence_lines, sentence_offset=sentence_offset,
                                                                     token_offset=token_offset)
            sentence_offset += 1
            token_offset += len(a_sentence.annotated_tokens)
            sentences.append(a_sentence)

        return sentences


def _find_value_in_lines(lines, key: str) -> str:
    ret = ""
    for line in lines:
        if line.find(key) == 0:
            pos = len(key)
            ret = line[pos:].strip()
    return ret


def _find_int_value_in_lines(lines, key: str) -> Optional[int]:
    for line in lines:
        if line.find(key) == 0:
            pos = len(key)
            ret = line[pos:].strip()
            if ret.isnumeric():
                return int(ret)
    return None
