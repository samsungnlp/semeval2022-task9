class Normalizer:

    def normalize(self, a_text: str) -> str:
        ret = a_text.lower()  # lowercase
        ret = ret.replace("\n", " ")  # build oneliner

        for c in ",.:-_!?/*()":
            ret = ret.replace(c, " ")  # remove excess of punctuation

        prev = ""
        while prev != ret:
            prev = ret
            ret = ret.replace("  ", " ")

        return ret
