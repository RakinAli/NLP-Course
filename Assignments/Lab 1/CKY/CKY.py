from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""


class CKY:

    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}}
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []

    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file):
        stream = open(grammar_file, mode="r", encoding="utf8")
        for line in stream:
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(",")
            if len(right) == 2:
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules:
                    first_rules = self.binary_rules[first]
                else:
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules:
                    second_rules = first_rules[second]
                    if left not in second_rules:
                        second_rules.append(left)
                else:
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1:
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules:
                    word_rules = self.unary_rules[word]
                    if left not in word_rules:
                        word_rules.append(left)
                else:
                    word_rules = [left]
                    self.unary_rules[word] = word_rules

    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s):
        """Produces a CKY parse table from the input sentence s."""
        """https://coli-saar.github.io/cl19/lectures/07-cky.pdf"""
        self.words = s.split()
        length = len(self.words)

        # Initialize the table with empty lists
        self.table = [[[] for _ in range(length + 1)] for _ in range(length)]
        self.backptr = [[[] for _ in range(length + 1)] for _ in range(length)]

        # Fill the diagonal of the table using unary rules --> Base case
        for i, word in enumerate(self.words):
            if word in self.unary_rules:
                self.table[i][i + 1] = self.unary_rules[word]

        # Fill in the table for phrases and sentences
        # Span size --> N 
        for span in range(2, length + 1):   
            # Start at second leftmost cell and move right
            # N size
            for start in range(length - span + 1):
                end = start + span
                # At most N size thus n^3
                for mid in range(start + 1, end):
                    # Check the combination of all entries in the left and right spans
                    for A in self.table[start][mid]:
                        for B in self.table[mid][end]:
                            if A in self.binary_rules:
                                if B in self.binary_rules[A]:
                                    # We have found a binary rule A -> B C
                                    C = self.binary_rules[A][B]
                                    self.table[start][end].extend(C)
                                    # Add backpointers
                                    for c in C:
                                        self.backptr[start][end].append((mid, A, B))

    # Prints the parse table
    def print_table(self):
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print(t.table)

    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees(self, row, column, symbol, top=True):
        print("DEBUGGER___________________________")
        # Print out the unary rules
        print("Unary rules: ", self.unary_rules)
        print("Binary rules: ", self.binary_rules)
        print("Table: ", self.table)
        print("Backpointers: ", self.backptr)
        print("Words: ", self.words)
        print("__________________________________")
        # Terminal case
        if row == column - 1:
            if symbol in self.unary_rules and self.words[row] in self.unary_rules[symbol]:
                print(f'{symbol}({self.words[row]})', end='')
            return

        # Non-terminal case
        if symbol not in self.table[row][column]:
            if top:
                print(f"No parse tree for symbol {symbol} at position {row}, {column}")
            return

        for sym, pair, k in self.backptr[row][column]:
            if sym == symbol:  # one of our guys
                if top:
                    print(f'{sym} ->', end=' ')
                print(f'(', end='')
                self.print_trees(row, k, pair[0], False)
                print(', ', end='')
                self.print_trees(k, column, pair[1], False)
                print(')', end='')
                if top:
                    print('')
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CKY parser")
    parser.add_argument(
        "--grammar",
        "-g",
        type=str,
        required=True,
        help="The grammar describing legal sentences.",
    )
    parser.add_argument(
        "--input_sentence",
        "-i",
        type=str,
        required=True,
        help="The sentence to be parsed.",
    )
    parser.add_argument(
        "--print_parsetable", "-pp", action="store_true", help="Print parsetable"
    )
    parser.add_argument("--print_trees", "-pt", action="store_true", help="Print trees")
    parser.add_argument("--symbol", "-s", type=str, default="S", help="Root symbol")

    arguments = parser.parse_args()

    cky = CKY(arguments.grammar)
    cky.parse(arguments.input_sentence)
    if arguments.print_parsetable:
        cky.print_table()
    if arguments.print_trees:
        cky.print_trees(len(cky.words) - 1, 0, arguments.symbol)


if __name__ == "__main__":
    main()
