# import argparse
# import datetime
from collections import namedtuple

import h5py

from dlgo import agent
from dlgo import scoring
from dlgo.goboard_fast import GameState, Player, Point


BOARD_SIZE = 9
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(BOARD_SIZE, 0, -1):
        line = []
        for col in range(1, BOARD_SIZE + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:BOARD_SIZE])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    print_board(game.board)
    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def main():         # 10.8
    # PLACEHOLDER VARIABLES (CHANGE BEFORE RUNNING SCRIPT)
    agent1filepath = ""
    agent2filepath = ""
    agent1 = agent.load_policy_agent(h5py.File(agent1filepath))
    agent2 = agent.load_policy_agent(h5py.File(agent2filepath))
    num_games = 50
    #

    wins = 0        # This script tracks wins and losses from the point of view of agent1
    losses = 0
    color1 = Player.black      # color1 = black, color2 = white
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            wins += 1
        else:
            losses += 1
        color1 = color1.other   # swap colors after each game, in case either agent plays better depending on color
    print('Agent 1 record: %d/%d' % (wins, wins + losses))


if __name__ == '__main__':
    main()
