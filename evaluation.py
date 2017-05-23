#!/usr/bin/env python3

"""
Waits for console input and prints the most likely language code
for each line entered.
"""
import requests

from charlm import CharLM
from identifier import LanguageIdentifier

def train(ngram_order, trainfiles):
    """
    Trains a character-level language model per language
    and adds these to a language identificator.
    """
    identifier = LanguageIdentifier()

    for language_code, training_data in trainfiles.items():
        print("Training {0} {1}-gram language model...".format(language_code, ngram_order))
        model = CharLM(ngram_order)
        model.train(training_data)
        identifier.add_model(language_code, model)
    return identifier

if __name__ == "__main__":

    trainfiles = {
        "CS": "data/newstest2009.cs",
        "DE": "data/newstest2009.de",
        "EN": "data/newstest2009.en",
        "ES": "data/newstest2009.es",
        "FR": "data/newstest2009.fr"
    }

    testfiles = {
        "DE": "data/donald.de",
        "EN": "data/donald.en",
        "FR": "data/donald.fr"
    }

    language_identifier_3gram = train(3, trainfiles)
    language_identifier_5gram = train(5, trainfiles)

    for lang_code, test_data in testfiles.items():
        print("______________________________________________")
        print(test_data + " is written in " + lang_code)
        print("______________________________________________")
        for sentence in CharLM.extract_data(test_data):

            result_3gram = language_identifier_3gram.identify(sentence)
            result_5gram = language_identifier_5gram.identify(sentence)

            sentence = sentence[:16]
            print('{sentence: <{width}} 3gram: {result3gram}, result5gram: {result5gram}'.format(sentence = sentence, width = 16, result3gram = result_3gram, result5gram = result_5gram))
