"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_x = sum(row.count(X) for row in board)
    num_o = sum(row.count(O) for row in board)

    if num_x <= num_o:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] is not EMPTY:
        raise Exception("Invalid move: position already taken")
    if i < 0 or i >= 3 or j < 0 or j >= 3:
        raise Exception("Invalid move: position out of bounds")
    new_board = [row[:] for row in board]
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for row in board:
        if all(cell == X for cell in row):
            return X
        elif all(cell == O for cell in row):
            return O

    for col in range(3):
        if all(board[row][col] == X for row in range(3)):
            return X
        elif all(board[row][col] == O for row in range(3)):
            return O

    # Check main diagonal
    if all(board[i][i] == X for i in range(3)):
        return X
    elif all(board[i][i] == O for i in range(3)):
        return O

    # Check secondary diagonal
    if all(board[i][2 - i] == X for i in range(3)):
        return X
    elif all(board[i][2 - i] == O for i in range(3)):
        return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    return all(all(cell is not EMPTY for cell in row) for row in board)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    result = winner(board)
    if result == X:
        return 1
    elif result == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def minimax_value(board, alpha, beta, maximizing):
        """
        Recursive helper function to determine the minimax value with alpha-beta pruning.
        """
        if terminal(board):
            return None, utility(board)

        if maximizing:
            best_value = -math.inf
            best_action = None
            for action in actions(board):
                new_board = result(board, action)
                _, value = minimax_value(new_board, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_action, best_value
        else:
            best_value = math.inf
            best_action = None
            for action in actions(board):
                new_board = result(board, action)
                _, value = minimax_value(new_board, alpha, beta, True)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_action, best_value

    current_player = player(board)
    best_action, _ = minimax_value(board, -math.inf, math.inf, current_player == X)
    return best_action
