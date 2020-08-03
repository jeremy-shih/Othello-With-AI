"""
An AI player for Othello. 
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

cached = {}

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    score = get_score(board)
    dark = score[0]
    light = score[1]

    if color == 1:
        utility = dark - light
    else:
        utility = light - dark

    return utility

# Better heuristic value of board
def compute_heuristic(board, color): #not implemented, optional
    #IMPLEMENT
    return 0 #change this!

############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):

    # if we are caching, check if in cached
    if caching and board in cached.keys():
        return cached[board]

    # if limit is 0, return none as best move, and score
    if limit == 0:
        return None, compute_utility(board, color)

    # find opponent's color
    if color == 1:
        opponent_color = 2
    else:
        opponent_color = 1

    # passed in is our AI's color, so get all possible moves from opponent's color with the new board that is passed in
    possible_moves = get_possible_moves(board, opponent_color)

    # if no moves left, return score
    if not possible_moves:
        return None, compute_utility(board, color)

    min_score = float("Inf")
    min_move = None

    for move in possible_moves:
        # get the new board from the move
        new_board = play_move(board, opponent_color, move[0], move[1])

        # for each move, recursively find max node using mini-max method, which will find min node, ...
        new_move, new_score = minimax_max_node(new_board, color, limit - 1, caching)

        # find the move with the lowest score
        if new_score < min_score:
            min_score = new_score
            min_move = move

        # add to cached if not inside already
        if caching:
            cached[new_board] = (min_move, min_score)

    return min_move, min_score



def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility

    # if we are caching, check if in cached
    if caching and board in cached.keys():
        return cached[board]

    # if limit is 0, return none as best move, and score
    if limit == 0:
        return None, compute_utility(board, color)

    # get all possible moves
    possible_moves = get_possible_moves(board, color)

    # if no moves left, return score
    if not possible_moves:
        return None, compute_utility(board, color)

    max_score = float("-Inf")
    max_move = None

    for move in possible_moves:
        # get the new board from the move
        new_board = play_move(board, color, move[0], move[1])

        # for each move, recursively find min node using mini-max method, which will find max node, ...
        new_move, new_score = minimax_min_node(new_board, color, limit - 1, caching)

        # find the move with the highest score
        if new_score > max_score:
            max_score = new_score
            max_move = move

        # add to cached if not inside already
        if caching:
            cached[new_board] = (max_move, max_score)

    return max_move, max_score

def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    # get best move according to the max score
    move, utility = minimax_max_node(board, color, limit, caching)

    # clear cache
    cached.clear()

    return move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):

    # if we are caching, check if in cached
    if caching and board in cached.keys():
        return cached[board]

    # if limit is 0, return none as best move, and score
    if limit == 0:
        return None, compute_utility(board, color)

    # find opponent's color
    if color == 1:
        opponent_color = 2
    else:
        opponent_color = 1

    # get all possible moves, AI color passed in, so feed in opponent's color for finding min
    possible_moves = get_possible_moves(board, opponent_color)

    # if no moves left, return score
    if not possible_moves:
        return None, compute_utility(board, color)

    min_score = float("Inf")
    min_move = None

    all_moves_sorted = []
    for move in possible_moves:
        # get the new board from the move, add to list for sorting
        new_board = play_move(board, opponent_color, move[0], move[1])
        all_moves_sorted.append((move, new_board))

    if ordering:
        # sort the moves by by score
        all_moves_sorted.sort(key=lambda input: compute_utility(input[1], color))

    # sorted moves are tuple of (move, new_board)
    for sorted_move in all_moves_sorted:
        move = sorted_move[0]
        new_board = sorted_move[1]

        # for each move, recursively find max node using alpha-beta method, which will find min node, and so on...
        new_move, new_score = alphabeta_max_node(new_board, color, alpha, beta, limit - 1, caching, ordering)

        # add to cache
        if caching:
            cached[new_board] = (new_move, new_score)

        # find min score and move
        if new_score < min_score:
            min_score = new_score
            min_move = move

        # prune if alpha >= beta
        if min_score <= alpha:
            return min_move, min_score

        # update beta value
        if min_score < beta:
            beta = min_score

    # After checking every move, return the minimum utility
    return min_move, min_score

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):

    # if we are caching, check if in cached
    if caching and board in cached.keys():
        return cached[board]

    # if limit is 0, return none as best move, and score
    if limit == 0:
        return None, compute_utility(board, color)

    # get all possible moves
    possible_moves = get_possible_moves(board, color)

    # if no moves left, return score
    if not possible_moves:
        return None, compute_utility(board, color)

    max_score = float("-Inf")
    max_move = None

    all_moves_sorted = []

    # moves are tuple of of possible coordinates
    for move in possible_moves:
        # get the new board from the move, add to list for sorting
        new_board = play_move(board, color, move[0], move[1])
        all_moves_sorted.append((move, new_board))

    # if we are using an ordering heuristic
    if ordering:
        # sort list by score (reversed to get largest to smallest score)
        all_moves_sorted.sort(key=lambda input: compute_utility(input[1], color), reverse=True)

    # sorted moves are tuple of (move, new_board)
    for sorted_move in all_moves_sorted:
        move = sorted_move[0]
        new_board = sorted_move[1]

        # for each move, recursively find min node using alpha-beta method, which will find max node, and so on...
        new_move, new_score = alphabeta_min_node(new_board, color, alpha, beta, limit - 1, caching, ordering)

        # add to cache
        if caching:
            cached[new_board] = (new_move, new_score)

        # find max score and move
        if new_score > max_score:
            max_score = new_score
            max_move = move

        # prune if alpha >= beta
        if max_score >= beta:
            return max_move, max_score

        # update alpha value
        if max_score > alpha:
            alpha = max_score

    return max_move, max_score


def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    # get max and min utility values
    alpha = float("-Inf")
    beta = float("Inf")

    # get best move according to the max score
    move, max_utility = alphabeta_max_node(board, color, alpha, beta, limit, caching, ordering)

    # clear cache
    cached.clear()

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
