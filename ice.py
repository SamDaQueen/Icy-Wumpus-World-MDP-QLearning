# "MDPs on Ice - Assignment 5"
# Ported from Java

import copy
import random
import sys

from numpy.random import choice

GOLD_REWARD = 100.0
PIT_REWARD = -150.0
DISCOUNT_FACTOR = 0.5
EXPLORE_PROB = 0.2  # for Q-learning
LEARNING_RATE = 0.1
ITERATIONS = 10000
MAX_MOVES = 1000
ACTIONS = 4
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
MOVES = ['U', 'R', 'D', 'L']

# Fixed random number generator seed for result reproducibility --
# don't use a random number generator besides this to match sol
random.seed(5100)


class Problem:
    # Problem class:  represents the physical space, transition probabilities, reward locations,
    # and approach to use (MDP or Q) - in short, the info in the text file
    # Fields:
    # approach - string, "MDP" or "Q"
    # move_probs - list of doubles, probability of going 1,2,3 spaces
    # map - list of list of strings: "-" (safe, empty space), "G" (gold), "P" (pit)

    # Format looks like
    # MDP    [approach to be used]
    # 0.7 0.2 0.1   [probability of going 1, 2, 3 spaces]
    # - - - - - - P - - - -   [space-delimited map rows]
    # - - G - - - - - P - -   [G is gold, P is pit]
    #
    # You can assume the maps are rectangular, although this isn't enforced
    # by this constructor.

    # __init__ consumes stdin; don't call it after stdin is consumed or outside that context
    def __init__(self):
        self.approach = input('Reading mode...')
        print(self.approach)
        probs_string = input("Reading transition probabilities...\n")
        self.move_probs = [float(s) for s in probs_string.split()]
        self.map = []
        for line in sys.stdin:
            self.map.append(line.split())

    def solve(self, iterations):
        if self.approach == "MDP":
            return mdp_solve(self, iterations)
        elif self.approach == "Q":
            return q_solve(self, iterations)
        return None


class Policy:
    # Policy: Abstraction on the best action to perform in each state - just a 2D string list-of-lists

    def __init__(self, problem):  # problem is a Problem
        # Signal 'no policy' by just displaying the map there
        self.best_actions = copy.deepcopy(problem.map)

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.best_actions])


def roll_steps(move_probs, row, col, move, rows, cols):
    # roll_steps:  helper for try_policy and q_solve -- "rolls the dice" for the ice and returns
    # the new location (r,c), taking map bounds into account
    # note that move is expecting a string, not an integer constant
    displacement = 1
    total_prob = 0
    move_sample = random.random()
    for p, prob in enumerate(problem.move_probs):
        total_prob += prob
        if move_sample <= total_prob:
            displacement = p+1
            break
    # Handle "slipping" into edge of map
    new_row = row
    new_col = col
    if not isinstance(move, str):
        print("Warning: roll_steps wants str for move, got a different type")
    if move == "U":
        new_row -= displacement
        if new_row < 0:
            new_row = 0
    elif move == "R":
        new_col += displacement
        if new_col >= cols:
            new_col = cols-1
    elif move == "D":
        new_row += displacement
        if new_row >= rows:
            new_row = rows-1
    elif move == "L":
        new_col -= displacement
        if new_col < 0:
            new_col = 0
    return new_row, new_col


def try_policy(policy, problem, iterations):
    # try_policy:  returns avg utility per move of the policy, as measured by "iterations"
    # random drops of an agent onto empty spaces, running until gold, pit, or time limit
    # MAX_MOVES is reached
    total_utility = 0
    total_moves = 0
    for i in range(iterations):
        # Resample until we have an empty starting square
        while True:
            row = random.randrange(0, len(problem.map))
            col = random.randrange(0, len(problem.map[0]))
            if problem.map[row][col] == "-":
                break
        for moves in range(MAX_MOVES):
            total_moves += 1
            policy_rec = policy.best_actions[row][col]
            # Take the move - roll to see how far we go, bump into map edges as necessary
            row, col = roll_steps(
                problem.move_probs, row, col, policy_rec,
                len(problem.map), len(problem.map[0]))
            if problem.map[row][col] == "G":
                total_utility += GOLD_REWARD
                break
            if problem.map[row][col] == "P":
                total_utility += PIT_REWARD
                break
    return total_utility / total_moves


def get_valid_actions(problem, row, col):
    # returns list of valid actions from a position on map

    actions = []
    if not col-1 < 0:
        actions.append(LEFT)
    if not col+1 > len(problem.map[0])-1:
        actions.append(RIGHT)
    if not row-1 < 0:
        actions.append(UP)
    if not row+1 > len(problem.map)-1:
        actions.append(DOWN)
    return actions


def get_expexted_utility(problem, row, col, utility):
    # returns a list of expected utilities of all possible successors
    # from a position with their respective probabilities

    expected_utility = []
    for action in get_valid_actions(problem, row, col):

        if action == LEFT:
            move = problem.move_probs[0]*utility[row][col-1]
            if not col-2 < 0:
                move += problem.move_probs[1]*utility[row][col-2]
            if not col-3 < 0:
                move += problem.move_probs[2]*utility[row][col-3]
            expected_utility.append(move)

        elif action == RIGHT:
            move = problem.move_probs[0]*utility[row][col+1]
            if not col+2 > len(problem.map[0])-1:
                move += problem.move_probs[1]*utility[row][col+2]
            if not col+3 > len(problem.map[0])-1:
                move += problem.move_probs[2]*utility[row][col+3]
            expected_utility.append(move)

        elif action == UP:
            move = problem.move_probs[0]*utility[row-1][col]
            if not row-2 < 0:
                move += problem.move_probs[1]*utility[row-2][col]
            if not row-3 < 0:
                move += problem.move_probs[2]*utility[row-3][col]
            expected_utility.append(move)

        elif action == DOWN:
            move = problem.move_probs[0]*utility[row+1][col]
            if not row+2 > len(problem.map)-1:
                move += problem.move_probs[1]*utility[row+2][col]
            if not row+3 > len(problem.map)-1:
                move += problem.move_probs[2]*utility[row+3][col]
            expected_utility.append(move)

    return expected_utility


def get_policy(problem, utility):
    # generates and returns policy using utility matrix

    row, col = len(problem.map), len(problem.map[0])
    policy = Policy(problem)

    # handle reward locations
    for r in range(row):
        for c in range(col):
            if problem.map[r][c] == 'G':
                utility[r][c] = GOLD_REWARD
            elif problem.map[r][c] == 'P':
                utility[r][c] = PIT_REWARD

    for r in range(row):
        for c in range(col):
            if policy.best_actions[r][c] == 'G'\
                    or policy.best_actions[r][c] == 'P':
                continue
            neighbors = {}
            # find max utility among neighbors, and find action for it
            for action in get_valid_actions(problem, r, c):
                if action == LEFT:
                    neighbors['L'] = utility[r][c-1]
                elif action == RIGHT:
                    neighbors['R'] = utility[r][c+1]
                elif action == UP:
                    neighbors['U'] = utility[r-1][c]
                elif action == DOWN:
                    neighbors['D'] = utility[r+1][c]
            policy.best_actions[r][c] = max(neighbors, key=neighbors.get)

    return policy


def mdp_solve(problem, iterations):
    # mdp_solve:  use [iterations] iterations of the Bellman equations over the whole map in [problem]
    # and return the policy of what action to take in each square

    row, col = len(problem.map), len(problem.map[0])

    # initialize reward with 0 for '-', and respective rewards for GOLD and PIT
    reward = [[0 for _ in range(col)] for _ in range(row)]
    for r in range(row):
        for c in range(col):
            if problem.map[r][c] == 'G':
                reward[r][c] = GOLD_REWARD
            elif problem.map[r][c] == 'P':
                reward[r][c] = PIT_REWARD

    utility = [[0 for _ in range(col)] for _ in range(row)]

    for _ in range(iterations):
        # create local matrix for utility for every iteration
        utility_local = [[0 for _ in range(col)] for _ in range(row)]
        for r in range(row):
            for c in range(col):
                # use bellman equation to update local utility matrix
                utility_local[r][c] = reward[r][c] + DISCOUNT_FACTOR * \
                    max(get_expexted_utility(problem, r, c, utility))
        # update global utility matrix
        utility = copy.deepcopy(utility_local)

    return get_policy(problem, utility)


def get_q_move(problem, r, c, q):
    # decide whether to select next move randomly or by best q
    randomise = int(choice(2, 1, p=[1-EXPLORE_PROB, EXPLORE_PROB]))
    if randomise:
        direction = choice(get_valid_actions(problem, r, c))
    else:
        valid = {}
        for action in get_valid_actions(problem, r, c):
            valid[action] = q[(r, c)][action]
        direction = max(valid, key=valid.get)

    move = 0
    if direction == UP:
        move = 'U'
    elif direction == RIGHT:
        move = 'R'
    elif direction == DOWN:
        move = 'D'
    elif direction == LEFT:
        move = 'L'
    # use roll_steps() to find successor move in direction
    return roll_steps(problem.move_probs, r, c, move,
                      len(problem.map), len(problem.map[0])), direction


def get_q_policy(problem, q):
    # for every state, select action with highest value in q-matrix
    policy = Policy(problem)
    row, col = len(problem.map), len(problem.map[0])
    for r in range(row):
        for c in range(col):
            if policy.best_actions[r][c] == 'G'\
                    or policy.best_actions[r][c] == 'P':
                continue
            valid = {}
            for action in get_valid_actions(problem, r, c):
                valid[action] = q[(r, c)][action]
            best_direction = max(valid, key=valid.get)
            if best_direction == UP:
                policy.best_actions[r][c] = 'U'
            elif best_direction == RIGHT:
                policy.best_actions[r][c] = 'R'
            elif best_direction == DOWN:
                policy.best_actions[r][c] = 'D'
            elif best_direction == LEFT:
                policy.best_actions[r][c] = 'L'

    return policy


def q_solve(problem, iterations):
    # using Q-learning to solve the problem

    row, col = len(problem.map), len(problem.map[0])

    reward = [[0 for _ in range(col)] for _ in range(row)]
    for r in range(row):
        for c in range(col):
            if problem.map[r][c] == 'G':
                reward[r][c] = GOLD_REWARD
            elif problem.map[r][c] == 'P':
                reward[r][c] = PIT_REWARD

    # initialize q(s, a) matrix
    q = {}
    for r in range(row):
        for c in range(col):
            q[(r, c)] = [0 for _ in range(ACTIONS)]

    for _ in range(iterations):
        # create new matrix for q for every iteration
        q_local = {}
        for x in range(row):
            for y in range(col):
                q_local[(x, y)] = [0 for _ in range(ACTIONS)]
        # start from a random non-final location at every iteration
        while True:
            r = random.randrange(0, len(problem.map))
            c = random.randrange(0, len(problem.map[0]))
            if problem.map[r][c] == "-":
                break
        # continue moving by best q value or randmoly (according to explore
        # probability) until gold or pit is found
        while True:
            if problem.map[r][c] == "G":
                for action in range(ACTIONS):
                    q[(r, c)][action] = GOLD_REWARD
                break
            elif problem.map[r][c] == "P":
                for action in range(ACTIONS):
                    q[(r, c)][action] = PIT_REWARD
                break
            # get next location and its direction using roll_steps()
            state, direction = get_q_move(problem, r, c, q)
            # use bellman equation for q-learning to update q-matrix
            q[(r, c)][direction] = (1-LEARNING_RATE)*q[(r, c)][direction] +\
                LEARNING_RATE * (reward[r][c] + DISCOUNT_FACTOR*max(q[state]))
            # set next location as current location
            r, c = state[0], state[1]

    return get_q_policy(problem, q)


# Main:  read the problem from stdin, print the policy and the utility over a test run
if __name__ == "__main__":
    problem = Problem()
    policy = problem.solve(ITERATIONS)
    print(policy)
    print("Calculating average utility...")
    print("Average utility per move: {utility:.2f}".format(
        utility=try_policy(policy, problem, ITERATIONS)))
