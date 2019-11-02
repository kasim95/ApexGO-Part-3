# 6.5, 6.6, and 6.7

import argparse
import numpy as np

from dlgo.encoders import get_encoder_by_name
from dlgo import goboard_fast as goboard
from dlgo import mcts
from utils import print_board, print_move


def generate_game(board_size, rounds, max_moves, temperature):
    # initialize encoded board state and encoded moves
    boards, moves = [], []

    # initialize a OnePlaneEncoder by name with given board size
    encoder = get_encoder_by_name('oneplane', board_size)

    # Instantiate a new game with board_size
    game = goboard.GameState.new_game(board_size)

    # MCTS agent bot with specified rounds and temp
    bot = mcts.MCTSAgent(rounds, temperature)

    num_moves = 0
    while not game.is_over():
        print_board(game.board)

        # bot picks next move
        move = bot.select_move(game)
        if move.is_play:
            # append encoded board to board
            boards.append(encoder.encode(game))

            # The one-hot-encoded next move is appended to moves
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)

        # apply bots move to the board
        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1

        # keep going until max number of moves is reached.
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60, help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-out')
    parser.add_argument('--move-out')

    # Customize through command line arguments
    args = parser.parse_args()
    xs = []
    ys = []

    for i in range(args.num_games):
        # Generate game data depending on number of games
        print('Generating game %d/%d...' % (i+1, args.num_games))
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)
        xs.append(x)
        ys.append(y)

    # Create labels after all games have been generated
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    # Save features and labels to separate files, specified by command line arguments

    np.save(args.board_out, x)
    np.save(args.move_out, y)


if __name__ == '__main__':
    main()
