import nltk
import os


def fetch_linguistic_resources() -> None:  # pragma: nocover
    nltk_resources = {
        "corpora": (
            "omw-1.4",
            "wordnet",
        ),
        "taggers": (
            "averaged_perceptron_tagger",
        ),
        "tokenizers": (
            "punkt",
        )
    }

    for category, resources in nltk_resources.items():
        for resource in resources:
            try:
                nltk.data.find(os.path.join(category, resource))
            except LookupError:
                nltk.download(resource)
