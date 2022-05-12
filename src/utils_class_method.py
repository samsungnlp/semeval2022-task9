class WordMistakesRepair:

    def __init__(self):
        self.strange_words = {
            "veggy": "veggie",
            "quinoum": "quinoa",
            "cocoum": "cocoa",
            "gnocchus": "gnocchi",
            "gremolatum": "gremolata",
            "boroccolus": "boroccoli",
            "muffulettum": "muffuletta",
            "lumaconus": "lumaconi",
            "guinness": "guinnes",
            "shred": "shre",
            "squares": "square",
            "halloumus": "halloumi",
            "molasses": "molass",
        }

        self.tricky_words = {
            "shred",
            "squares"
        }

    def change_question(self, question):
        for key, value in self.strange_words.items():
            if key not in self.tricky_words:
                question = question.replace(key, value) if key in question else question

        return question

    def change_answer(self, question: str, answer: str) -> str:
        for key, value in self.strange_words.items():
            if key in question.replace("?", "").replace(",", "").split() and value in answer.replace(",", "").split():
                answer = answer.replace(value, key)
        return answer
