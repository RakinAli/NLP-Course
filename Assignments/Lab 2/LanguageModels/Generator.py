import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""
# Used to print out statements and for debugging purposes
debugger = True

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):

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

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 

        grab_key = self.index[w]
        sentence = []
        
        sentence.append(w)
        
        for _ in range(n):
            word_combo = []
            word_prob = []
            for keys in self.bigram_prob:
                # Grab all the keys that have the same first word
                if keys[0] == grab_key:
                    word_combo.append(keys)
                    easier_prob = math.exp(self.bigram_prob[keys])
                    word_prob.append(easier_prob)

            if len(word_combo) != 0:
                # Pick according to the distribution
                next_word = random.choices(population=word_combo, weights=word_prob)[0]
                # Convert number to word
                next_id = next_word[1]
                next_token = self.word[next_id]
                sentence.append(next_token)
                grab_key = next_id
            else:
                # If the word is not in the model, pick a random word
                next_id = random.randint(0, self.unique_words-1)
                next_token = self.word[next_id]
                sentence.append(next_token)
                grab_key = next_id

        print(" ".join(sentence))

def main():
    """
    Parse command line arguments
    """
    if debugger:
        """
        Parse command line arguments
        """
        arguments = argparse.Namespace()
        arguments.file = 'kafka_model.txt'
        arguments.start = "gregor"
        arguments.number_of_words = 10

        generator = Generator()
        generator.read_model(arguments.file)
        generator.generate(arguments.start,arguments.number_of_words)

    else:
        parser = argparse.ArgumentParser(description='BigramTester')
        parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
        parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
        parser.add_argument('--number_of_words', '-n', type=int, default=100)

        arguments = parser.parse_args()

        generator = Generator()
        generator.read_model(arguments.file)
        generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
