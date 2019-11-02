import random
from dlgo.agent import Agent
from dlgo.gotypes import Player

MIN_SCORE, MAX_SCORE = -999999, 999999


def alpha_beta_result(game_state, max_depth, best_black, best_white, eval_fn):
    if game_state.is_over():
        return MAX_SCORE if game_state.winner() == game_state.next_player else MIN_SCORE

    if max_depth == 0:
        return eval_fn(game_state)

    best_so_far = MIN_SCORE

    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)

        opponent_best_result = alpha_beta_result(next_state, max_depth - 1, best_black, best_white, eval_fn)
        best_so_far = max(best_so_far, -1 * opponent_best_result)

        if game_state.next_player == Player.white:
            if best_so_far > best_white:
                best_white = best_so_far

            outcome_for_black = -1 * best_so_far

            # this outcome is worse than a better known outcome --> no need to investigate this leaf or subtree any more
            if outcome_for_black < best_black:
                return best_so_far
        elif game_state.next_player == Player.black:
            if best_so_far > best_black:
                best_black = best_so_far

            outcome_for_white = -1 * best_so_far

            # similar reasoning as above
            if outcome_for_white < best_white:
                return best_so_far

    return best_so_far


class AlphaBetaAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        super().__init__()

        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        best_black, best_white = MIN_SCORE, MIN_SCORE

        # all possible moves
        for possible_move in game_state.legal_moves():
            # game state if this move were selected
            next_state = game_state.apply_move(possible_move)

            opponent_best_outcome = alpha_beta_result(next_state, self.max_depth, best_black, best_white, self.eval_fn)
            our_best_outcome = -opponent_best_outcome

            if (not best_moves) or our_best_outcome > best_score:
                # best move so far
                best_moves = [possible_move]
                best_score = our_best_outcome

                if game_state.next_player == Player.black:
                    best_black = best_score
                elif game_state.next_player == Player.white:
                    best_white = best_score

            elif our_best_outcome == best_score:
                # as good as all currently known best moves
                best_moves.append(possible_move)

        return random.choice(best_moves)
