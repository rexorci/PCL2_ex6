#!/usr/bin/env python3

from math import pow, log2
from collections import defaultdict

class CharLM:
    """Implements a character-level n-gram language model."""

    BOS_SYMBOL = object()
    EOS_SYMBOL = object()

    def __init__(self, n=3):
        """Initialises a language model of order @param n."""
        self._order = n
        self._logprobs = defaultdict(lambda: defaultdict(float))

    @staticmethod
    def log(probability):
        """Transforms @param probability into log space."""
        return log2(probability)

    @staticmethod
    def perplexity(log_probability, n_items):
        """Returns the perplexity of a sequence with @param n_items and @param log_probability."""
        # entropy is average negative log2 likelihood per word
        entropy = -log_probability / n_items
        # perplexity is 2^entropy
        perplexity = pow(2, entropy)
        return perplexity

    def _extract_ngrams(self, sentence):
        """Returns the n-grams contained in @param sentence."""
        # beginning of sentence
        symbols = [self.BOS_SYMBOL] * (self._order-1)
        # actual characters; unknowns are not replaced (see exercise sheet)
        symbols += [char for char in sentence]
        # end of sentence
        symbols += [self.EOS_SYMBOL]
        # n-gram extraction, as in https://goo.gl/91x6P6
        return list(zip(*[symbols[i:] for i in range(self._order)]))

    def _add_ngram(self, head, history, log_probability):
        """
        Adds an n-gram to this language model, such that
        P(@param symbol|@param history) = @param log_probability.
        """
        self._logprobs[history][head] = log_probability

    def _set_unk_given_unknown_history(self, log_probability):
        """
        Sets the log probability used for n-grams with a history we
        have not seen in training.
        """
        self._logprobs.default_factory = lambda: defaultdict(lambda: log_probability)

    def _set_unk_given_known_history(self, history, log_probability):
        """
        Sets the log probability used for n-grams with a history we
        have seen in training, but not in combination with the current
        head.
        """
        self._logprobs[history].default_factory = lambda: log_probability

    def train(self, training_data):
        """
        Trains this language model on the sentences contained in
        file @param training_data (one sentence per line).
        """
        #TODO: Count n-gram occurrences, calculate their
        #      probability, and store them via self._add_ngram().
        #
        #      Use self._extract_ngrams() to extract ngrams from
        #      sentences.

    def get_perplexity(self, sentence):
        """Returns the perplexity of @param sentence."""
        log_probability = 0.0
        for ngram in self._extract_ngrams(sentence):
            head, history = ngram[-1], ngram[:-1]
            log_probability += self._logprobs[history][head]
        # +1 in length for EOS_SYMBOL (see PCL2 Session 10, slide 48)
        return self.perplexity(log_probability, len(sentence)+1)
