#!/usr/bin/env python3

"""
Waits for console input and prints the most likely language code
for each line entered.
"""

from charlm import CharLM
from identifier import LanguageIdentifier

def train():
    """
    Trains a character-level language model per language
    and adds these to a language identificator.
    """
    ngram_order = 3
    identifier = LanguageIdentifier()
    for language_code, training_data in {
            "CS": "data/newstest2009.cs",
            "DE": "data/newstest2009.de",
            "EN": "data/newstest2009.en",
            "ES": "data/newstest2009.es",
            "FR": "data/newstest2009.fr"
    }.items():
        print("Training {0} language model...".format(language_code))
        model = CharLM(ngram_order)
        model.train(training_data)
        identifier.add_model(language_code, model)
    return identifier

if __name__ == "__main__":
    language_identifier = train()
    while True:
        print("\nType in a sentence in one of the following languages: {0}\n"
              .format(", ".join(language_identifier.get_languages())))
        print(language_identifier.identify(input()))
