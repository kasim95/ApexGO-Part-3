import random
import time
import dlgo.zero as zero

import h5py

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo import scoring
from dlgo.utils import print_board


def run(board_size, first, second):
    start = time.time()

    # black_agent = random.choice([first, second])
    # white_agent = first if black_agent is second else first
    black_agent = first
    white_agent = second

    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    game_state = GameState.new_game(board_size)
    next_move = None

    while not game_state.is_over() and (next_move is None or not next_move.is_pass):  # random bot too stupid to stop the game
        move_timer_start = time.time()
        next_move = agents[game_state.next_player].select_move(game_state)

        if game_state.next_player is Player.black and next_move.is_pass:
            next_move = agents[game_state.next_player].select_move(game_state)

        print(f'{game_state.next_player} made move in {time.time() - move_timer_start} s')
        print(f'{game_state.next_player} selected {next_move}')

        game_state = game_state.apply_move(next_move)

        print(chr(27) + '[2J')  # clears board
        print_board(game_state.board)

        print('Estimated result: ')
        print(scoring.compute_game_result(game_state))

    print(f'Finished game in {time.time() - start} s')
    game_result = scoring.compute_game_result(game_state)

    print(game_result)

    first_won = False

    if game_result.winner == Player.black:
        if black_agent is first:
            first_won = True
    else:
        if white_agent is first:
            first_won = True

    if first_won:
        print("First agent wins!")
    else:
        print("Second agent wins!")


if __name__ == '__main__':

    with h5py.File('agz_bot.h5', 'r') as bot_file:
        bot1 = zero.load_zero_agent(bot_file)

    # with h5py.File('agz_bot_i1_2stone_hc.h5', 'r') as bot_file:
    #     bot2 = zero.load_zero_agent(bot_file)

    import dlgo.agent
    bot2 = dlgo.agent.naive.RandomBot()

    bot1.num_rounds = bot2.num_rounds = 1000
    #bot1.c = bot2.c = 0.5

    run(19, bot1, bot2)
