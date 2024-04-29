import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
import sys
import random


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
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    @property
    def vocab_size(self):
        return self.__V

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list

        :param      line:  The line
        :type       line:  str
        """
        #
        # REPLACE WITH YOUR CODE HERE
        #
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
        context = []

        # Calculate the range of indices for the right and left context
        right_context_indices = range(i + 1, min(i + 1 + self.__rws, len(sent)))
        left_context_indices = range(i - 1, max(i - 1 - self.__lws, -1), -1)

        # Collect indices for the right context words
        context.extend(self.__w2i.get(sent[idx], -1) for idx in right_context_indices)

        # Collect indices for the left context words and append them
        context.extend(self.__w2i.get(sent[idx], -1) for idx in left_context_indices)

        return context

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
        #
        # REPLACE WITH YOUR CODE
        #
        self.__w2i = {}
        self.__i2w = []

        focus_words = []
        context_words = []

        self.__total_words = 0
        self.__unigram = {}
        self.__V = 0

        line_gen = self.text_gen()
        for sentence in line_gen:
            # 1) Build the maps between words and indexes and vice versa
            self.build_mappings(sentence)
            # 3) Focus and a context words
            for focus_idx, focus_word in enumerate(sentence):
                focus_words.append(focus_word)
                context_words.append(self.get_context(sentence, focus_idx))

        self.create_unigram_dist()
        self.create_corrected_unigram_dist()

        # Create a txt file with focus words
        with open("my_focus_words.txt", "w") as f:
            for word in focus_words:
                f.write(word + "\n")

        # Create a txt file with context words
        # Print context as txt
        with open("my_context.txt", "w") as f:
            for context in context_words:
                f.write(f"{context}\n")

        return focus_words, context_words 

    def create_corrected_unigram_dist(self):
        denominator = 0
        self.__corrected_unigram = self.__unigram.copy()
        for word in self.__unigram:
            denominator += self.__unigram[word]**0.75
        for word in self.__corrected_unigram:
            self.__corrected_unigram[word] = self.__unigram[word]**0.75 / denominator

    def create_unigram_dist(self):
        for unique_word in self.__unigram:
            self.__unigram[unique_word] = self.__unigram[unique_word] / self.__total_words

    def build_mappings(self, sentences):
        for word in sentences:
            self.__total_words += 1
            if word not in self.__unigram:
                self.__V += 1
                self.__unigram[word] = 1
                self.__w2i[word] = len(self.__i2w)
                self.__i2w.append(word)
            else:
                self.__unigram[word] += 1

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.

        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        # Initialize the set of taboo indices
        taboo_indices = {xb, pos}

        # Select the appropriate word list based on the flag
        word_list = self.__corrected_unigram if self.__use_corrected else self.__unigram

        # Filter out taboo words
        available_words = [word for idx, word in enumerate(word_list.keys()) if idx not in taboo_indices]

        # Sample negative examples from the available words
        if len(available_words) >= number:
            negative_samples = random.sample(available_words, number)
        else:
            # If not enough available words, fall back to less restrictive sampling
            negative_samples = random.choices(available_words, k=number)

        return negative_samples

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION

        # Normalised initialization
        self.__W = np.random.normal(0, 0.1, (self.__V, self.__H)) # Word embeddings
        self.__U = np.random.normal(0, 0.1, (self.__V, self.__H)) # Negative samples

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                words_seen = i + ep * N
                focus_word = x[i]
                focus_word_ind = self.__w2i[focus_word]

                # Getting positive context words -- context words
                pos_sample_indices = t[i]
                pos_sample_words = [self.__i2w[word] for word in pos_sample_indices]

                # Getting negative context words
                negative_samples = self.generate_neg_sample(
                    focus_word_ind, pos_sample_indices
                )

                # Do gradient descent
                self.gradient_descent(focus_word_ind, pos_sample_words, negative_samples, words_seen)

    def gradient_descent(self, focus_word_ind, positive_samples, negative_samples, words_seen):
        """
        This code does Gradient descent accordingly to the slides from the lecture. 
        """

        # Derivative with respect to the focus vector
        focus_grads = self.grads_wrt_focusVec(focus_word_ind, positive_samples, negative_samples)

        # Derivative with respect to the negative samples
        negative_grads = self.compute_neg_grads(focus_word_ind, negative_samples)

        # Derivative with respect to the positive samples
        positive_grads = self.compute_pos_grads(focus_word_ind, positive_samples)

        # Update the focus vector
        self.__W[focus_word_ind] -= self.__lr * focus_grads

        # Update the negative samples
        self.update_sample_vectors(negative_grads)
        self.update_sample_vectors(positive_grads)

        #Update the learning rate
        self.update_learning_rate(words_seen)

    def update_learning_rate(self, words_seen):
        """
        Update the learning rate according to the formula in the assignment
        """ 
        if self.__lr < self.__init_lr * 0.0001:
            self.__lr = self.__init_lr * 0.0001
            return
        else:
            self.__lr = self.__init_lr * (1 - words_seen / (self.__epochs * len(self.__i2w)+1))



    def update_sample_vectors(self, gradients):
        """
        Update the vectors of the samples in the weight matrix U
        """
        for word_ind, gradient in gradients.items():
            self.__U[word_ind] -= self.__lr * gradient        


    def compute_pos_grads(self, focus_word_idx, pos_samples):
        # Get the focus vector
        focus_vec = self.__W[focus_word_idx]

        # Gram the indices of positive samples
        pos_indices = [self.__w2i[ps] for ps in pos_samples]

        # Retrieve all positive sample vectors at once using advanced indexing
        pos_vectors = self.__U[pos_indices]  

        # Compute dot products of the focus vector with all positive vectors
        dot_products = np.dot(pos_vectors, focus_vec)  # Shape: (num_pos_samples,)

        # Compute the sigmoid of these dot products, then subtract 1
        sigmoid_minus_one = self.sigmoid(dot_products) - 1  

        
        # We multiply each (sigmoid - 1) result with the focus vector to get the gradients
        gradients = np.outer(sigmoid_minus_one, focus_vec)  # Shape: (num_pos_samples, self.__H)

        # Create a dictionary mapping from indices to gradient vectors
        pos_samples_gradients_dict = {idx: gradients[i] for i, idx in enumerate(pos_indices)}

        return pos_samples_gradients_dict


    def compute_neg_grads(self, focus_word_idx, neg_samples):
        # Retrieve the focus vector from the weight matrix using the focus word index
        focus_vec = self.__W[focus_word_idx]

        # Convert negative sample word tokens into their corresponding indices
        neg_indices = [self.__w2i[ns] for ns in neg_samples]

        # Retrieve all negative sample vectors at once using advanced indexing
        neg_vectors = self.__U[neg_indices]  # Shape: (num_neg_samples, self.__H)

        # Compute dot products of the focus vector with all negative vectors
        dot_products = np.dot(neg_vectors, focus_vec)  # Shape: (num_neg_samples,)

        # Compute the sigmoid of these dot products
        sigmoids = self.sigmoid(dot_products)  # Shape: (num_neg_samples,)

        # We multiply each sigmoid result with the focus vector to get the gradients
        # We need to make sure the result is aligned properly in terms of shape
        # Broadcasting the focus vector across all rows of neg_vectors
        gradients = np.outer(sigmoids, focus_vec) 

        # Create a dictionary mapping from indices to gradient vectors
        neg_samples_gradients_dict = {
            idx: gradients[i] for i, idx in enumerate(neg_indices)
        }
        return neg_samples_gradients_dict

    def grads_wrt_focusVec(self, focus_word_ind, positive_samples, negative_samples):
        # Get the focus word vector
        focus_vec = self.__W[focus_word_ind]

        # Get vectors for positive and negative samples using vectorized indexing
        pos_indices = [self.__w2i[ps] for ps in positive_samples]
        neg_indices = [self.__w2i[ns] for ns in negative_samples]

        pos_vectors = self.__U[pos_indices]  # Shape: (num_pos_samples, self.__H)
        neg_vectors = self.__U[neg_indices]  # Shape: (num_neg_samples, self.__H)

        # Compute dot products in a vectorized manner
        pos_dot_products = np.dot(pos_vectors, focus_vec)  # Shape: (num_pos_samples,)
        neg_dot_products = np.dot(neg_vectors, focus_vec)  # Shape: (num_neg_samples,)

        # Compute gradients for positive and negative samples
        pos_gradients = pos_vectors * (self.sigmoid(pos_dot_products)[:, np.newaxis] - 1)
        neg_gradients = neg_vectors * self.sigmoid(neg_dot_products)[:, np.newaxis]

        # Sum gradients from positive and negative samples
        focus_gradient = np.sum(pos_gradients, axis=0) + np.sum(neg_gradients, axis=0)

        return focus_gradient

    def generate_neg_sample(self, focus_word_ind, positive_sample_indices):
        negative_samples = []
        for pos_ind in positive_sample_indices:
            negative_samples.extend(self.negative_sampling(self.__nsample, focus_word_ind, pos_ind))
        return negative_samples

    def find_nearest(self, words, k=5, metric='cosine'):
        # Your code here - done
        nearest_words = []

        nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(self.__matrix)

        for word in words:
            context_vector_word = self.get_word_vector(word)
            if context_vector_word is None:
                continue
            distances, indices = nn.kneighbors([context_vector_word])

            clostest_words = []
            for i in range(len(indices[0])):
                index = indices[0][i]
                distance = distances[0][i]
                the_word = self.__words[index]
                clostest_words.append((the_word, distance))
            nearest_words.append(clostest_words)

        return nearest_words

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
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
