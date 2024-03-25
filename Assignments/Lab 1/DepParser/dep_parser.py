from pathlib import Path
from parse_dataset import Dataset
import argparse


class Parser:
    SH, LA, RA = 0, 1, 2

    def conllu(self, source):
        buffer = []
        for line in source:
            line = line.rstrip()  # strip off the trailing newline
            if not line.startswith("#"):
                if not line:
                    yield buffer
                    buffer = []
                else:
                    columns = line.split("\t")
                    if columns[0].isdigit():  # skip range tokens
                        buffer.append(columns)

    def trees(self, source):
        """
        Reads trees from an input source.

        Args: source: An iterable, such as a file pointer.

        Yields: Triples of the form `words`, `tags`, heads where: `words`
        is the list of words of the tree (including the pseudo-word
        <ROOT> at position 0), `tags` is the list of corresponding
        part-of-speech tags, and `heads` is the list of head indices
        (one head index per word in the tree).
        """
        for rows in self.conllu(source):
            words = ["<ROOT>"] + [row[1] for row in rows]
            tags = ["<ROOT>"] + [row[3] for row in rows]
            tree = [0] + [int(row[6]) for row in rows]
            relations = ["root"] + [row[7] for row in rows]
            yield words, tags, tree, relations

    def step_by_step(self, string):
        """
        Parses a string and builds a dependency tree. In each step,
        the user needs to input the move to be made.
        """
        w = ("<ROOT> " + string).split()
        i, stack, pred_tree = 0, [], [0] * len(w)  # Input configuration
        while True:
            print("----------------")
            print("Buffer: ", w[i:])
            print("Stack: ", [w[s] for s in stack])
            print("Predicted tree: ", pred_tree)
            try:
                ms = input("Move: (Shift,Left,Right): ").lower()[0]
                m = (
                    Parser.SH
                    if ms == "s"
                    else Parser.LA if ms == "l" else Parser.RA if ms == "r" else -1
                )
                if m not in self.valid_moves(i, stack, pred_tree):
                    print("Illegal move")
                    continue
            except:
                print("Illegal move")
                continue
            i, stack, pred_tree = self.move(i, stack, pred_tree, m)
            if i == len(w) and stack == [0]:
                # Terminal configuration
                print("----------------")
                print("Final predicted tree: ", pred_tree)
                return

    def create_dataset(self, source, train=False):
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(source) as f:
            for w, tags, tree, relations in self.trees(f):
                i, stack, pred_tree = 0, [], [0] * len(tree)  # Input configuration
                m = self.compute_correct_move(i, stack, pred_tree, tree)
                while m != None:
                    ds.add_datapoint(w, tags, i, stack, m, train)
                    i, stack, pred_tree = self.move(i, stack, pred_tree, m)
                    m = self.compute_correct_move(i, stack, pred_tree, tree)
        return ds

    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.

        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.

        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        moves = []

        # YOUR CODE HERE

        # As long as there are words that have not been processed yet, we can shift
        if i < len(pred_tree):
            moves.append(self.SH)

        # Create an arc from the second topmost word to the topmost word on the stack, so we need at least two words on the stack
        if i > 2 and len(stack) >= 2:
            moves.append(self.RA)

        # Create an arc from the topmost word to the second topmost word on the stack then remove the second topmost word from the stack. The length of the stack must be >= 3
        if i > 2 and len(stack) >= 2: 
            moves.append(self.LA)

        return moves

    def move(self, i, stack, pred_tree, move):
        """
        Executes a single move.

        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.

        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """

        # YOUR CODE HERE

        # If the move is shift, we add the index of the first unprocessed word to the stack and increment i
        if move == self.SH:
            stack.append(i)
            i += 1

        # If the move is left arc,
        elif move == self.LA:
            topmost = stack[-1]
            second_topmost = stack[-2]
            # Arc from the topmost word to the second topmost word on the stack
            pred_tree[second_topmost] = topmost
            # Second topmost word from the stack gone
            stack.pop(-2)

        elif move == self.RA:
            second_topmost = stack[-2]
            topmost = stack[-1]
            # Arc from the second topmost word to the topmost word on the stack
            pred_tree[topmost] = second_topmost
            # Topmost word from the stack gone
            stack.pop(-1)

        return i, stack, pred_tree

    def compute_correct_moves(self, tree):
        """
        Computes the sequence of moves (transformations) the parser
        must perform in order to produce the input tree.
        """
        i, stack, pred_tree = 0, [], [0] * len(tree)  # Input configuration
        moves = []
        m = self.compute_correct_move(i, stack, pred_tree, tree)
        while m != None:
            moves.append(m)
            i, stack, pred_tree = self.move(i, stack, pred_tree, m)
            m = self.compute_correct_move(i, stack, pred_tree, tree)
        return moves

    def compute_correct_move(self, i, stack, pred_tree, correct_tree):
        """
        Given a parser configuration (i,stack,pred_tree), and 
        the correct final tree, this method computes the  correct 
        move to do in that configuration.
    
        See the textbook 18.2.1. 
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            correct_tree: The correct dependency tree.
        
        Returns:
            The correct move for the specified parser
            configuration, or `None` if no move is possible.
        """
        """
        The algorithm is as follows:
        1. Check for only shift condition
        2. Create a copy of the predicted tree and try LEFT ARC (LA)
        3. Create a copy of the predicted tree and try RIGHT ARC (RA)
        4. Decide on LEFT ARC or RIGHT ARC based on conditions
        5. For left arc the condition is that the number of matching pairs should be greater than the current number of matching pairs
        6. For right arc the condition is that the number of matching pairs should be greater than the current number of matching pairs and all dependents of the topmost word are assigned and match the correct tree
        7. Default to SHIFT if none of the above conditions are met
        # The idea was taken by looking at the textbook then copy pasting psuedo code in chatgpt3 then modifying it to fit the code
        """
        # Atleast have the same length
        assert len(pred_tree) == len(correct_tree)

        # Early SHIFT condition
        if i < 2 and len(stack) < 2:
            return self.SH

        #Baseline to compare with 
        numMatchingPairs = sum(a == b for a, b in zip(pred_tree, correct_tree))

        if len(stack) >= 2:
            topmost_word = stack[-1]
            second_topmost_word = stack[-2]

            # Try LEFT ARC (LA)
            la_pred_tree = pred_tree.copy()
            la_pred_tree[second_topmost_word] = topmost_word
            la_increase = sum(a == b for a, b in zip(la_pred_tree, correct_tree)) > numMatchingPairs

            # Try RIGHT ARC (RA)
            ra_pred_tree = pred_tree.copy()
            ra_pred_tree[topmost_word] = second_topmost_word
            ra_increase = sum(a == b for a, b in zip(ra_pred_tree, correct_tree)) > numMatchingPairs
            # Special case: all dependents of the topmost word are assigned and match the correct tree 
            ra_all_dependents_assigned = ra_pred_tree.count(topmost_word) == correct_tree.count(topmost_word)

            # Decide on LEFT ARC or RIGHT ARC based on conditions
            if la_increase:
                return self.LA
            elif ra_increase and ra_all_dependents_assigned:
                return self.RA
            elif pred_tree == correct_tree:  # Special case for completing the tree
                return self.RA

        # Default to SHIFT if none of the above conditions are met
        return self.SH if i < len(pred_tree) else None


filename = Path("en-ud-dev-projective.conllu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transition-based dependency parser")
    parser.add_argument(
        "-s", "--step_by_step", type=str, help="step-by-step parsing of a string"
    )
    parser.add_argument(
        "-m",
        "--compute_correct_moves",
        type=str,
        default=filename,
        help="compute the correct moves given a correct tree",
    )
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step(args.step_by_step)

    elif args.compute_correct_moves:
        with open(args.compute_correct_moves, encoding="utf-8") as source:
            for w, tags, tree, relations in p.trees(source):
                print(p.compute_correct_moves(tree))
