import nltk


class PuttyLemmatizer:
    """
    a lemmatizer with putty-fallback
    """

    def __init__(self):
        """
        Please add known cases
        """
        self.verb_exceptions = {
            "blent": "blend",
            "lied": "lay",
            "prehet": "preheat",
            "plated": "plate",
            "shre": "shred",
        }
        self.noun_exceptions = {
            "chilli": "chili",
            "chillies": "chili",
            "chilly": "chili",
            "chily": "chili",
            "chilies": "chili",
            "chiles": "chili",
            "veggy": "veggie",
            "veggies": "veggie",
            "quinoum": "quinoa",
            "cocoum": "cocoa",
            "fettuccinus": "fettuccini",
            "frittatum": "frittata",
            "gnocchus": "gnocchi",
            "gremolatum": "gremolata",
            "korvgrytum": "korvgryta",
            "krispie": "krispies",
            "krispy": "krispies",
            "smarty": "smarties",
            "patty": "patties",
            "cofee": "coffee",
            "yogurt": "yoghurt",
            "towl": "towel",
            "suger": "sugar",
            "parley": "parsley",
            "filet": "fillet",
            "cardamom": "cardamon",
            "boroccoli": "broccoli",
            "boroccolus": "broccoli",
            "hommas": "hummus",
            "muffulettum": "muffuletta",
            "lumaconus": "lumaconi",
        }
        self.lemmatizer = nltk.WordNetLemmatizer()

    def lemmatize_verb(self, verb: str) -> str:
        return self.verb_exceptions.get(verb, self.lemmatizer.lemmatize(verb, "v"))

    def lemmatize_noun(self, noun: str) -> str:
        return self.noun_exceptions.get(noun, self.lemmatizer.lemmatize(noun, "n"))
