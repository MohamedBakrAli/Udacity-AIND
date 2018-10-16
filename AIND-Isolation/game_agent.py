"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # get the number of moves that i could do
    my_moves = len(game.get_legal_moves(player))
    # get the number of moves that my opponent could do
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(my_moves - opponent_moves)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # get the number of moves that i could do
    my_moves = len(game.get_legal_moves(player))
    # get the number of moves that my opponent could do
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(my_moves -  (2 *opponent_moves))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # get the number of moves that i could do
    my_moves = len(game.get_legal_moves(player))
    # get the number of moves that my opponent could do
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if (my_moves == 0):
        return float("inf")

    if (opponent_moves == 0):
        return float("-inf")

    return float(my_moves / opponent_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        # check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # max-value function (helper function)
        def max_value(self, game, depth):

            # check for timeout
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # get the valid move for this player
            valid_moves = game.get_legal_moves()

            # intial the best value with inf to can get the max_value
            best_value = float("-inf")

            # terminal state if reach the end of the depth or have no valid move
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)

            for m in valid_moves:
                best_value = max(best_value, min_value(self, game.forecast_move(m), depth-1))
            return best_value

        # min-value function (helper function)
        def min_value(self, game, depth):
            # check for timeout
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # get the valid move for this player
            valid_moves = game.get_legal_moves()

            # intial the best value with inf to can get the min_value
            best_value = float("inf")

            # terminal state if reach the end of the depth or have no valid move
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)

            for m in valid_moves:
                best_value = min(best_value, max_value(self, game.forecast_move(m), depth-1))
            return best_value

        # get the legel move for this player
        valid_moves = game.get_legal_moves()
        # int best_move, best_score
        best_score = float("-inf")
        best_move = (-1, -1)

        # check for terminal state
        if (not valid_moves):
            return best_move
        # check for all legel move to get the best one
        for m in valid_moves:
            score = min_value(self, game.forecast_move(m), depth - 1)
            if score > best_score:
                best_score = score
                best_move = m

        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)

        # Check if my agent is first to move
        if (game._board_state[-1] == -1):
            if (game.get_legal_moves()):
                best_move = (4, 4)
            return best_move
        try :
            # Iterative Deepning, stop when timeout
            depth = 0
            while (True) :
                depth += 1
                best_move = self.alphabeta(game, depth)

        except SearchTimeout :
            # Handle any actions required after timeout as needed
            pass

        return best_move
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        # check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # max-value function (helper function)
        def max_value(self, game, depth, alpha, beta):
            # check for timeout
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            # get the legel moves
            valid_moves  = game.get_legal_moves()
            # intial the best_move by - inf to can update to the max value
            best_move = float("-inf")
            # check for the terminal state
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)
            # loop for the valid_moves to get the best move
            for m in valid_moves:
                best_move = max(best_move , min_value(self, game.forecast_move(m), depth-1, alpha, beta))
                # check for purning when best bossible value is equal or higher than beta
                if (best_move >= beta):
                    return best_move
                # Update alpha if best possible value is higher than alpha
                alpha = max(best_move , alpha)
            return best_move

        # min-value function (helper function)
        def min_value(self, game, depth, alpha, beta):
            # check for timeout
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            # get the legel moves
            valid_moves  = game.get_legal_moves()
            # intial the best_move by  + inf to can update to the min value
            best_move = float("inf")
            # check for the terminal state
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)
            # loop for the valid_moves to get the best move
            for m in valid_moves:
                best_move = min(best_move , max_value(self, game.forecast_move(m), depth-1, alpha, beta))
                # check for purning when best bossible value is equal or higher than alpha
                if (best_move <= alpha):
                    return best_move
                # Update beta if best possible value is higher than beta
                beta = min(best_move , beta)
            return best_move
        # Main alphabeta function
        # get legel moves
        valid_moves = game.get_legal_moves()
        # intial best_move, best_score
        best_move = (-1, -1)
        best_score = float("-inf")
        # check for terminal state
        if (depth == 0) or (not valid_moves):
            return best_move

        for m in valid_moves:
            v = min_value(self, game.forecast_move(m), depth -1, alpha, beta)
            if (v > best_score):
                best_score = v
                best_move = m
                alpha = max(alpha, v)
                if (best_score >= beta):
                    return best_move

        return best_move
