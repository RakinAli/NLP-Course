from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye. Modified 2024 by me 
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
    #       """Produces a CKY parse table from the input sentence s."""
    """https://coli-saar.github.io/cl19/lectures/07-cky.pdf"""


    def parse(self, s):
        self.words = s.split()
        length = len(self.words)

        # Initialize the table with empty lists, adjusted to not include the first empty column
        self.table = [[[] for _ in range(length)] for _ in range(length)]
        self.backptr = [[[] for _ in range(length)] for _ in range(length)]

        # Fill the diagonal (i, i) cells of the table using unary rules for single words
        for i, word in enumerate(self.words):
            if word in self.unary_rules:
                for rule in self.unary_rules[word]:
                    self.table[i][i].append(
                        rule
                    )  # Note the change here from i][i + 1] to i][i]

        # Fill in the table for spans larger than 1
        for span in range(2, length + 1):
            for start in range(length - span + 1):
                end = start + span - 1  # Adjusted to work with the updated table dimensions
                for mid in range(start, end):  # Note: mid goes from start to end - 1
                    for A in self.table[start][mid]:
                        for B in self.table[mid + 1][end]:
                            if A in self.binary_rules and B in self.binary_rules[A]:
                                for C in self.binary_rules[A][B]:
                                    if C not in self.table[start][end]:
                                        self.table[start][end].append(C)
                                        self.backptr[start][end].append((mid, A, B))
        # Prints the parse table
    def print_table(self):
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print(t.table)

    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'

    def print_trees(self, row, column, symbol, prefix=""):
        # Base case: if we are at a leaf node in the parse tree (only unary rules).
        if row == column:  
            # Check if the symbol matches any of the unary rules for the word at this position.
            if symbol in self.unary_rules and self.words[row] in self.unary_rules[symbol]:
                print(f"{prefix}{symbol} -> '{self.words[row]}'")
            return

        # If the symbol is not in the current cell, there is no parse tree starting with this symbol here.
        if symbol not in self.table[row][column]:
            return

        # Check each backpointer entry to see if it can produce the required symbol.
        for mid, left_symbol, right_symbol in self.backptr[row][column]:
            # Only print the rules if the current backpointer's left and right symbols correspond to the current cell.
            if left_symbol in self.binary_rules and right_symbol in self.binary_rules[left_symbol]:
                if symbol in self.binary_rules[left_symbol][right_symbol]:
                    # Print the current production rule.
                    print(f"{prefix}{symbol} -> {left_symbol} {right_symbol}")
                    # Recursively print the parse trees for the left and right parts of the rule.
                    self.print_trees(row, mid, left_symbol, prefix + "  ")
                    self.print_trees(mid + 1, column, right_symbol, prefix + "  ")

        # Additionally, handle the case where a unary rule may apply to a non-terminal.
        if symbol in self.unary_rules:
            for unary_producer in self.unary_rules[symbol]:
                # This checks if a unary rule can be applied to any of the symbols in the current cell.
                if unary_producer in self.table[row][column]:
                    print(f"{prefix}{symbol} -> {unary_producer}")
                    self.print_trees(row, column, unary_producer, prefix + "  ")


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
        cky.print_trees(0, len(cky.words) - 1, arguments.symbol)


if __name__ == "__main__":
    main()
