#!/usr/bin/env python3

from charlm import CharLM

class LanguageIdentifier:
    """
    Identifies the language used in any string.

    A LanguageIdentifier stores 1..* character-level language
    models (`CharLM` objects). Given a string, it returns the language code associated with the most suitable model, i.e.,
    the model with the lowest perplexity for that string.
    """

    def __init__(self):
        self._models = {}

    def get_languages(self):
        """Returns the language codes this identifier can handle."""
        return list(self._models.keys())

    def add_model(self, language_code, model):
        """Adds an language model to this identifier."""
        if language_code in self._models:
            raise Warning("Already defined language code {0}. Current model will be replaced.".format(language_code))
        self._models[language_code] = model

    def identify(self, sentence):
        """
        Returns the most likely language used in @param
        sentence, given the models stored in this identifier.
        """
        assert len(self._models) > 2, "At least two models are needed for language identification."
        perplexities = {
            language_code: model.get_perplexity(sentence) for \
            language_code, model in self._models.items()
        }
        return sorted(perplexities, key=perplexities.get)[0]
