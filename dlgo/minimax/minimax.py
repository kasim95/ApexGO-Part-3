import enum
import random
from dlgo.agent import Agent
from dlgo.goboard_fast import GameState


class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3


def reverse_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    elif game_result == GameResult.win:
        return game_result.loss
    else:
        return game_result.draw


def best_result(game_state: GameState):
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return GameResult.win
        elif game_state.winner() is None:
            return GameResult.draw
        else:
            return GameResult.loss

    best_result_so_far = GameResult.loss

    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        opponent_best_result = best_result(next_state)
        our_result = reverse_game_result(opponent_best_result)

        best_result_so_far = max(our_result, best_result_so_far)

    return best_result_so_far


class MinimaxAgent(Agent):
    def select_move(self, game_state: GameState):
        winning_moves = []
        draw_moves = []
        losing_moves = []

        # loop through all legal moves:
        for move in game_state.legal_moves():
            # state of the game after this move is applied
            next_state = game_state.apply_move(move)

            # determine opponent's best outcome given that state
            opponent_best_outcome = best_result(next_state)

            our_best_outcome = reverse_game_result(opponent_best_outcome)

            if our_best_outcome == GameResult.win:
                winning_moves.append(move)
            elif our_best_outcome == GameResult.draw:
                draw_moves.append(move)
            else:
                losing_moves.append(move)

        # try to win, with drawing the next best choice
        if winning_moves or draw_moves:
            return random.choice(winning_moves or draw_moves)

        # lost the game
        return random.choice(draw_moves)
