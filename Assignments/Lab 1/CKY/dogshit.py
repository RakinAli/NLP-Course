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
                        second_rules.append[left]
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
        self.words = s.split()

        # initialise the table
        for _ in range(len(self.words)):
            self.table.append([[] for _ in range(len(self.words))])
            self.backptr.append([{} for _ in range(len(self.words))])

        # fill the diagonal
        for i in range(len(self.words)):
            self.table[i][i] = self.unary_rules[self.words[i]]

        # fill the rest of the table from down to up
        for j in range(1, len(self.words)):
            for i in range(j - 1, -1, -1):
                possibilities = []
                for k in range(i, j):
                    # print("i", i, "j", j, "k", k)
                    left = self.table[i][k]
                    right = self.table[k + 1][j]
                    if left != [] and right != []:
                        combination = [[l, r] for l in left for r in right]
                        # print(left, "+", right, "-->", combination)
                        for c in combination:
                            try:
                                result = self.binary_rules[c[0]][c[1]][0]
                                possibilities.append(result)
                                if result not in self.backptr[i][j]:
                                    self.backptr[i][j][result] = [
                                        [i, k, c[0], k + 1, j, c[1]]
                                    ]
                                else:
                                    self.backptr[i][j][result].append(
                                        [i, k, c[0], k + 1, j, c[1]]
                                    )
                            except:
                                continue
                self.table[i][j] = possibilities
        # for i in range(len(self.backptr)):
        #     for j in range(len(self.backptr[0])):
        #         if self.backptr[i][j] != {}:
        #             print(i, j, self.backptr[i][j])

    # Prints the parse table
    def print_table(self):
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print(t.table)

    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees(self, row, column, symbol):
        tree = self.rec_print_trees(row, column, symbol)
        print(tree)

    def rec_print_trees(self, row, column, symbol):
        if row == column:
            return symbol + "(" + self.words[row] + ")"

        tree = ""
        prev_combinations = self.backptr[column][row][symbol]
        for prev_c in prev_combinations:
            left_tree = self.rec_print_trees(prev_c[1], prev_c[0], prev_c[2])
            right_tree = self.rec_print_trees(prev_c[4], prev_c[3], prev_c[5])
            if "\n" not in left_tree and "\n" not in right_tree:
                tree += symbol + "(" + left_tree + ", " + right_tree + ")"
                if len(prev_combinations) > 1:
                    tree += "\n"
            else:
                left_tree_list = left_tree.split("\n")
                for lt in left_tree_list:
                    right_tree_list = right_tree.split("\n")
                    for rt in right_tree_list:
                        if lt != "" and rt != "":
                            tree += symbol + "(" + lt + ", " + rt + ")\n"

        # check redundancy
        trees = set(tree.split("\n"))
        tree = ""
        for t in trees:
            if t != "":
                tree += t + "\n"

        return tree


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
