#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

    def read_model(self,filename):
        """
            Reads the contents of the language model file into the appropriate data structures.

            :param filename: The name of the language model file.
            :return: <code>true</code> if the entire file could be processed, false otherwise.
            """
        # Taken from generate.py
        # Your code here
        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                # Reads the first line, extracts
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # Handling the V lines
                for _ in range(self.unique_words):
                    id, word, count = f.readline().strip().split(' ')
                    id, count = map(int, (id, count))
                    # Number to word
                    self.index[word] = id
                    # Word to number
                    self.word[id] = word
                    self.unigram_count[word] = count

                # Rest of the lines
                rest = f.readlines()
                biograms = len(rest) -1 # No Biogram on the last one 
                for i in range(biograms):
                    word1, word2, log_prob = rest[i].strip().split(' ')
                    word1, word2 = map(int, (word1, word2))
                    self.bigram_prob[(word1, word2)] = float(log_prob)
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def compute_entropy_cumulatively(self, word):
        """
        -1/N * sum(Log(P(w_i-1, w_i))) where n is the number of tokens in the test corpus

        To handle missing words, and missing bigrams we use linear interpolation

        P(w_i-1, w_i) = lambda1 * P(w_i| w_i-1) + lambda2 * P(w_i) + lambda3 

        Values of lambdas are given in the assignment

        """
        # First word in the test corpus
        if self.test_words_processed == 0:
            current_word_index = self.index[word] if word in self.index else -1
            cumulative_log_probability = 0

        # Rest words
        elif self.test_words_processed >= 1:
            current_word_index = self.index[word] if word in self.index else -1
            previous_word_index = self.last_index

            # Handling different cases
            if current_word_index == -1:
                # Case 1: Both current and previous words are unknown
                cumulative_log_probability = self.lambda3
                # Case 2: Previous word is unknown but current word is known
            elif previous_word_index == -1:
                # Case 2: Previous word is unknown
                cumulative_log_probability = (
                    self.lambda2 * self.unigram_count[word] / self.total_words
                    + self.lambda3
                )
            elif previous_word_index != -1 and current_word_index != -1:
                # Case 3: Both current and previous words are known
                if (previous_word_index, current_word_index) in self.bigram_prob:
                    bigram_probability = self.lambda1 * math.exp(
                        self.bigram_prob[(previous_word_index, current_word_index)]
                    )
                    unigram_probability = (
                        self.lambda2 * self.unigram_count[word] / self.total_words
                    )
                    cumulative_log_probability = (
                        bigram_probability + unigram_probability + self.lambda3
                    )
                else:
                    unigram_probability = (
                        self.lambda2 * self.unigram_count[word] / self.total_words
                    )
                    cumulative_log_probability = unigram_probability + self.lambda3

            # Update cumulative log probability based on test words processed
            if self.test_words_processed == 1:
                self.logProb = (
                    -1 / self.test_words_processed * math.log(cumulative_log_probability)
                )

            elif self.test_words_processed > 1:
                self.logProb = (
                    self.logProb * -(self.test_words_processed - 1)
                    + math.log(cumulative_log_probability)
                ) / -self.test_words_processed

        # Update state variables
        self.test_words_processed += 1
        self.last_index = current_word_index

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                # Here's the for loop that goes through the tokens
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
