import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

import re
from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling

    def init_params(self, W, w2i, i2w):
        # __w is the word embeddings, __w2i is a dictionary mapping words to their index in the vocabulary
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    @property
    def vocab_size(self):
        return self.__V

    def clean_line(self, line):
        # YOUR CODE HERE - Done

        # Could not find the python library function to remove
        return re.sub(r"\d|[^\w\s]", "", line).lower().split()

    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices

        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """

        # Extracting left and right window sizes from class attributes
        left_window_size = self.__lws
        right_window_size = self.__rws

        # Retrieving the length of the sentence
        sent_length = len(sent)

        # Constructing a list of context indices within the sentence
        context_indices_in_sent = list(range(i - left_window_size, i)) + list(
            range(i + 1, i + right_window_size + 1)
        )

        # Ensuring that context indices are within the bounds of the sentence
        context_indices_in_sent = [
            item for item in context_indices_in_sent if 0 <= item < sent_length
        ]

        # Extracting context words from the sentence based on context indices
        context_words = [sent[index] for index in context_indices_in_sent]

        # Getting the respective context indices in the entire corpus
        context_indices = [self.w2i[word] for word in context_words]

        return context_indices

    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
        (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        # REPLACE WITH YOUR CODE

        """
        A function preparing data for a skipgram word2vec model.
        """
        self.w2i = {}
        self.i2w = {}
        self.unigram_count = {}
        self.__V = 0  # Vocabulary size
        self.__TotalWords = 0  # Total number of words in the corpus

        index = 0
        for line in self.text_gen():
            for word in line:
                self.__TotalWords += 1
                if word not in self.w2i:
                    self.w2i[word] = index
                    self.i2w[index] = word
                    self.unigram_count[word] = 1
                    self.__V += 1  # Increment vocabulary size only for new words
                    index += 1
                else:
                    self.unigram_count[word] += 1

        """
        # Optionally, write the w2i dictionary to a text file
        with open("w2i.txt", 'w') as f:
            for key, value in self.w2i.items():
                f.write('%s:%s\n' % (key, value))
        """

        # Calculate the unigram distribution
        self.unigram_distribution = {}
        total_words = sum(self.unigram_count.values())
        for word in self.unigram_count:
            self.unigram_distribution[word] = self.unigram_count[word] / total_words

        # Calculate the corrected unigram distribution
        """
        Slide 40 Lecture 6: P_s(w) = P_unigram(w)^0.75 / sum(P_unigram(w)^0.75)

        Basically sum the denom
        """
        self.corrected_unigram = {}
        denominator = 0
        for word in self.unigram_distribution:
            denominator += self.unigram_distribution[word] ** 0.75
        for word in self.unigram_distribution:
            self.corrected_unigram[word] = self.unigram_distribution[word] ** 0.75 / denominator

        # Step 3: Return two two lists: Focus words and context words
        focus_words = []
        context_indices = []
        for line in self.text_gen():
            for i, word in enumerate(line):
                if word not in focus_words:
                    focus_words.append(word)
                    context_indices.append(self.get_context(line, i))
                if word in focus_words:
                    focus_index = focus_words.index(word)
                    context_indices[focus_index].extend(self.get_context(line, i))

        # Print w2i as txt
        with open("w2i.txt", 'w') as f:
            for key, value in self.w2i.items():
                f.write('%s:%s\n' % (key, value))

        return focus_words, context_indices

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    import numpy as np

    def create_alias_table(self,probs):
        n = len(probs)
        scaled_probs = np.array(probs) * n
        aliases = np.zeros(n, dtype=np.int32)
        small = []
        large = []
        
        for i, prob in enumerate(scaled_probs):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        while small and large:
            small_index = small.pop()
            large_index = large.pop()
            
            aliases[small_index] = large_index
            scaled_probs[large_index] = (scaled_probs[large_index] - 1) + (scaled_probs[small_index] - 1)
            
            if scaled_probs[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)
        
        return scaled_probs, aliases


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples avoiding the words in `xb` and `pos` words.
        """
        use_corrected = self.__use_corrected
        unigram = self.corrected_unigram if use_corrected else self.unigram_distribution

        negative_samples_indices = []
        keys = list(unigram.keys())
        probabilities = np.array(list(unigram.values()))
        probabilities /= probabilities.sum()  # Normalize probabilities

        while len(negative_samples_indices) < number:
            candidates = np.random.choice(keys, size=number*2, p=probabilities)
            for candidate in candidates:
                candidate_index = self.w2i[candidate]
                if candidate_index not in negative_samples_indices and candidate_index != xb and candidate_index != pos:
                    negative_samples_indices.append(candidate_index)
                if len(negative_samples_indices) == number:
                    break

        return negative_samples_indices

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        self.__W = np.random.rand(self.__V, self.__H)
        self.__U = np.random.rand(self.__V, self.__H)

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):

                # YOUR CODE HERE

                # Get the focus word and its index
                focus_word = x[i]
                focus_word_index = self.w2i[focus_word]
                # Get the context words and their indices
                postive_samples_indices = t[i]
                postive_samples = [self.i2w[index] for index in postive_samples_indices]

                # Generate negative samples
                negative_samples_indices = self.generate_neg_samples(focus_word_index, postive_samples_indices)

                negative_samples = [self.i2w[index] for index in negative_samples_indices]

                self.gradient_descent(
                    focus_word_index,
                    postive_samples,
                    negative_samples,
                    i,
                    N,
                    ep,
                )

    def generate_neg_samples(self, focus_word_idx, pos_sample_dix):
        neg_samples = []
        for pos_sample_ind in pos_sample_dix:
            neg_samples.extend(self.negative_sampling(self.__nsample, focus_word_idx, pos_sample_ind))
        return neg_samples

    def gradient_descent(self, focus_word_idx, pos_sample_dix, neg_sample_dix, current_words, total_words, total_epochs):

        # Compute gradient with respect to the focus word
        focus_word_gradient = np.zeros(self.__H)
        focus_vec = self.__W[focus_word_idx]

        for postive_sample in pos_sample_dix:
            positive_vec = self.__U[self.w2i[postive_sample]]
            focus_word_gradient += self.sigmoid(np.dot(focus_vec, positive_vec)) * positive_vec

        for negative_sample in neg_sample_dix:
            negative_vec = self.__U[self.w2i[negative_sample]]
            focus_word_gradient += self.sigmoid(np.dot(focus_vec, negative_vec)) * negative_vec

        # Compute gradient with respect to the positive samples
        positive_grads_dic = {}
        focus_vec = self.__W[focus_word_idx]
        for positive_sample in pos_sample_dix:
            positive_vec = self.__U[self.w2i[positive_sample]]
            positive_grads_dic[self.w2i[positive_sample]] = focus_vec * (self.sigmoid(np.dot(positive_vec, focus_vec))-1)

        # Compute gradient with respect to the negative samples
        negative_grads_dic = {}
        focus_vec = self.__W[focus_word_idx]
        for negative_sample in neg_sample_dix:
            negative_vec = self.__U[self.w2i[negative_sample]]
            negative_grads_dic[self.w2i[negative_sample]] = self.sigmoid(np.dot(focus_vec, negative_vec)) * focus_vec

        # Update the focus word vector
        self.__W[focus_word_idx] -= self.__lr * focus_word_gradient

        # Update the positive and negative samples vectors
        for positive_sample in pos_sample_dix:
            self.__U[self.w2i[positive_sample]] -= self.__lr * positive_grads_dic[self.w2i[positive_sample]]

        for negative_sample in neg_sample_dix:
            self.__U[self.w2i[negative_sample]] -= self.__lr * negative_grads_dic[self.w2i[negative_sample]]

        # Update the learning rate
        if self.__lr < self.__init_lr * 0.0001:
            self.__lr = self.__init_lr * 0.0001
        else:
            self.__lr = self.__init_lr * (1 - current_words / (total_words + 1 * total_epochs))

    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
        [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #
        return []

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                i=0
                for i, w in enumerate(self.__i2w):
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
                    i+=1
        except:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
