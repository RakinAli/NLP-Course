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
    """https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/12_slides-2x2.pdf"""

    def parse(self, s):
        self.words = s.split()
        print("Words:", self.words)
        print("Unary rules:", self.unary_rules)
        print("Binary rules:", self.binary_rules)
    

        # Init the table and backptr as 2 dimensional arrays
        for _ in range(len(self.words)):
            self.table.append([[] for _ in range(len(self.words))])
            self.backptr.append([{} for _ in range(len(self.words))])

        # fill the diagonal
        for i in range(len(self.words)):
            self.table[i][i] = self.unary_rules[self.words[i]]

        # fill the rest of the table. Left to right
        for end in range(1, len(self.words)):
            for start in range(end - 1, -1, -1):
                possibilities = []
                # Bottom to top however diagonally from left to right
                for k in range(start, end):
                    left = self.table[start][k]
                    right = self.table[k + 1][end]
                    if left != [] and right != []:
                        combination = [[l, r] for l in left for r in right]
                        for c in combination:
                            try:
                                result = self.binary_rules[c[0]][c[1]][0]
                                possibilities.append(result)
                                # If the result is not in the backpointer, add it
                                if result not in self.backptr[start][end]:
                                    self.backptr[start][end][result] = [
                                        [start, k, c[0], k + 1, end, c[1]]
                                    ]
                                else:
                                    self.backptr[start][end][result].append(
                                        [start, k, c[0], k + 1, end, c[1]]
                                    )
                            except:
                                continue
                self.table[start][end] = possibilities

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
