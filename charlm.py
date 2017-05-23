#!/usr/bin/env python3

from math import pow, log2
from collections import defaultdict

from pandas import DataFrame


class CharLM:
    """Implements a character-level n-gram language model."""

    BOS_SYMBOL = object()
    EOS_SYMBOL = object()

    #A set of all characters of that language
    V = set()

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

    @staticmethod
    def _extract_data(data_file):
        """extract the given data file"""
        extracted_file = open(data_file, 'r', encoding='utf-8')
        extracted_file = extracted_file.readlines()
        return extracted_file

    def p_laplace(self, head_count, history_count):
        """
        
        :return: 
        """
        return self.log((head_count + 1) / (history_count + len(self.V)))

    def train(self, training_data):
        """
        Trains this language model on the sentences contained in
        file @param training_data (one sentence per line).
        """

        # Count n-gram occurrences, calculate their
        # probability, and store them via self._add_ngram().
        # Use self._extract_ngrams() to extract ngrams from
        # sentences.

        sentences = CharLM._extract_data(training_data)

        # We extract the ngrams for each sentence using the provided function,
        # then we store the number of occurencies in a nested dictionary.
        ngram_counts = defaultdict(lambda: defaultdict(int))

        for sentence in sentences:
            self.V |= set(sentence)
            for ngram in self._extract_ngrams(sentence):
                head, history = ngram[-1], ngram[:-1]
                ngram_counts[history][head] += 1

        self._set_unk_given_unknown_history(self.p_laplace(0,0))

        for history, heads in ngram_counts.items():
            history_count = sum(heads.values())
            self._set_unk_given_known_history(history, self.p_laplace(0,history_count))

            for head, head_count in heads.items():
                self._add_ngram(head, history, self.p_laplace(head_count, history_count))

    def get_perplexity(self, sentence):
        """Returns the perplexity of @param sentence."""
        log_probability = 0.0
        for ngram in self._extract_ngrams(sentence):
            head, history = ngram[-1], ngram[:-1]
            log_probability += self._logprobs[history][head]
        # +1 in length for EOS_SYMBOL (see PCL2 Session 10, slide 48)
        return self.perplexity(log_probability, len(sentence)+1)
