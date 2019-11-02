import random
from dlgo.agent import Agent

MIN_SCORE, MAX_SCORE = -999999, 999999


def best_result(game_state, max_depth, eval_fn):
    if game_state.is_over():
        return MAX_SCORE if game_state.winner() == game_state.next_player else MIN_SCORE

    if max_depth == 0:
        return eval_fn(game_state)

    best_so_far = MIN_SCORE

    for candidate in game_state.legal_moves():
        next_state = game_state.apply_move(candidate)
        opponent_best_result = best_result(next_state, max_depth - 1, eval_fn)
        best_so_far = max(best_so_far, -1 * opponent_best_result)

    return best_so_far


class DepthPrunedAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        super().__init__()

        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None

        # loop over all possible moves given this state
        for possible_move in game_state.legal_moves():
            next_state = game_state.apply_move(possible_move)

            # given this new game state, figure out opponent's best outcome
            opponent_best_outcome = best_result(next_state, self.max_depth, self.eval_fn)

            our_best_outcome = -1 * opponent_best_outcome

            if (not best_moves) or our_best_outcome > best_score:
                # best move so far
                best_moves = [possible_move]
                best_score = our_best_outcome
            elif our_best_outcome == best_score:
                # just as good as currently known best
                best_moves.append(possible_move)

        return random.choice(best_moves)
