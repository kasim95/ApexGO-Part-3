import random
import numpy as np

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.goboard_slow import Move
from dlgo.gotypes import Point


class RandomBot(Agent):
    def select_move(self, game_state):
        """Choose a random valid move that preserves our own eyes"""
        candidates = []

        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)

                if game_state.is_valid_move(Move.play(candidate)) and \
                    not is_point_an_eye(game_state.board,
                                        candidate,
                                        game_state.next_player):
                    candidates.append(candidate)

        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))


class FastRandomBot(Agent):
    def __init__(self):
        super().__init__()
        self.dim = None
        self.point_cache = []

    def _update_cache(self, dim):
        self.dim = dim
        rows, cols = dim
        self.point_cache = []
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                self.point_cache.append(Point(row=row, col=col))

    def select_move(self, game_state):
        # choose a move that preserves own eyes
        dim = (game_state.board.num_rows, game_state.board.num_cols)

        if dim != self.dim:
            self._update_cache(dim)

        idx = np.arange(len(self.point_cache))
        np.random.shuffle(idx)

        for i in idx:
            p = self.point_cache[i]
            if game_state.is_valid_move(Move.play(p)) and \
                    not is_point_an_eye(game_state.board, p, game_state.next_player):
                return Move.play(p)

        return Move.pass_turn()  # can't make a move that doesn't ruin own eyes
