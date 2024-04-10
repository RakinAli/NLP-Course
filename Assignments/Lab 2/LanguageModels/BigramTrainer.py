#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import faulthandler
faulthandler.enable()
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD2417 Language Engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell. Modified 2024 by Rakin to solve the assignment.
"""
# Used to print out statements and for debugging
debugger = False

"""
______________________ Vid redo av uppgiften ______________________
Last index: Pekare till ordet BAKOM!!!
Unique words: Antal unika ord i träningskorpora
Total words: Antal ord i träningskorpora
Unigram count: Antal gånger ett ord förekommer i träningskorpora
Bigram count: Antal gånger ett bigram förekommer i träningskorpora
Index: Ord till siffra
Word: Siffra till ord
"""

class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file f.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = text_file.read().encode('utf-8').decode().lower()
        try :
            self.tokens = nltk.word_tokenize(text) 
            if debugger:
                print("Token: ", self.tokens)
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE
        """
        ALGORITHM:
        For unigram:
        If word is not in index: You need to convert word into a number and store it in index.
        If word is in index: You need to update the count of the word in unigram_count.

        For bigram:
        If last_index is -1 then we have not processed any word yet. So, we do not have any bigram to update. We just update the last_index to the current word index after processing the word.

        If last_index is not -1 then we have processed a word before. So, we have a bigram to update. We update the bigram count of the bigram (last_index, current word index) by 1.        
        """
        # For unigram
        if token not in self.index:
            # Mapping from words to identifiers
            self.index[token] = self.unique_words
            # Mapping from identifiers to words
            self.word[self.unique_words] = token

            # Update the unigram counts
            self.unigram_count[token] += 1
            self.unique_words += 1
            self.total_words += 1
        else:
            # Update the unigram counts
            self.unigram_count[token] += 1
            self.total_words += 1

        # For bigram
        if self.last_index != -1:
            if (self.last_index ,self.index[token]) in self.bigram_count:
                self.bigram_count[(self.last_index,self.index[token])] += 1
            else:
                self.bigram_count[(self.last_index,self.index[token])] = 1

        # Update the lagging pointer to the current token for the next iteration
        self.last_index = self.index[token]    

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []

        # The first line
        first_line = str(self.unique_words) + ' ' + str(self.total_words)
        rows_to_print.append(first_line)

        # All the V lines: Identifier, word, unigram count
        for i in range(self.unique_words):
            index_number = str(i)
            word = self.word[i]
            unigram_count = str(self.unigram_count[word])
            row = index_number + ' ' + word + ' ' + unigram_count
            rows_to_print.append(row)

        # Identifier of word 1 and word 2 then probability of bigram
        # Apparently should be sorted 
        sort_bigram = sorted(self.bigram_count.items())
        """ 
        Quick maths:
        P(w2|w1) = count(w1,w2)/count(w1). 
        If biogram (4,6) then count(4,6) = 2 and count(4) = 3. after P(w2|w1) = 2/3
        """
        for biogram,freq in sorted(self.bigram_count.items()): 
            token_1 = str(biogram[0])
            token_2 = str(biogram[1])
            # Marginal probability of the first token in the bigram
            Marginal = self.unigram_count[self.word[biogram[0]]] 
            if Marginal != 0:
                bigram_probability_ln = format(math.log(freq/Marginal),'.15f') 
                    
            else:
                bigram_probability_ln = format(0,'.15f')
            
            probs = str(bigram_probability_ln)
            row = token_1 + ' ' + token_2 + ' ' + probs
            rows_to_print.append(row)

            # The final line
        rows_to_print.append(str(-1))

        return rows_to_print

    def __init__(self):
        """
        Constructor. Processes the file f and builds a language model
        from it.

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed. Pekare till föregående ord som behandlats.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    if debugger:
        print("Index: ", bigram_trainer.index, "\n")
        print("Word: ", bigram_trainer.word ,"\n")
        print("Unigram Count: ", bigram_trainer.unigram_count,"\n")
        print("Bigram Count: ", bigram_trainer.bigram_count , "\n")
        print("Unique Words: ", bigram_trainer.unique_words, "\n")
        print("Total Words: ", bigram_trainer.total_words, "\n")




    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


if __name__ == "__main__":
    main()
