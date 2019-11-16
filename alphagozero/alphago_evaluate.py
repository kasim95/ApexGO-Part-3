import os
import sys
import concurrent.futures
import time
import random

from keras import backend as K

import h5py
import dlgo.zero as zero
from dlgo.networks.zero import zero_model

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo import scoring


# def evaluate_network(known_best, learner, board_size, num_games=100, ratio=0.55):
#     known_best_wins = 0
#
#     for game_idx in range(num_games):
#         # learner needs to win at least 55% of the time
#         # Therefore, if known best has won so many games that this is
#         # impossible, exit early
#         if known_best_wins / num_games >= 0.45:
#             return known_best
#         elif (game_idx - known_best_wins) / num_games >= 0.55:
#             # for similar reasoning, we might be able to determine the learner is better early
#             return learner
#
#         black_agent = random.choice([known_best, learner])
#         white_agent = known_best if black_agent is learner else learner
#
#         agents = {
#             Player.black: black_agent,
#             Player.white: white_agent,
#         }
#
#         game = GameState.new_game(board_size)
#
#         while not game.is_over():
#             next_move = agents[game.next_player].select_move(game)
#             game = game.apply_move(next_move)
#
#         game_result = scoring.compute_game_result(game)
#
#         print(f"game {game_idx + 1} result: {game_result}")
#
#         if game_result.winner == Player.black:
#             if black_agent is known_best:
#                 known_best_wins += 1
#         else:
#             if white_agent is known_best:
#                 known_best_wins += 1
#
#     if known_best_wins / num_games >= 0.45:  # learner won < 0.55%, so not measurably stronger
#         return known_best
#     else:
#         return learner


def simulate():
    assert os.path.exists('agz_bot_train.h5')

    # load known best bot
    if os.path.exists('agz_bot.h5'):
        with h5py.File('agz_bot.h5', 'r') as best_bot:
            best_agent = zero.load_zero_agent(best_bot)

    else:
        return True  # learner bot wins! ... by default, since there is no best currently

    # load learner bot
    with h5py.File('agz_bot_train.h5') as learn_bot:
        learner_agent = zero.load_zero_agent(learn_bot)

    # randomly decide first move
    black_agent = random.choice([best_agent, learner_agent])
    white_agent = best_agent if black_agent is learner_agent else learner_agent

    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    game = GameState.new_game((learner_agent.encoder.board_size, learner_agent.encoder.board_size))

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    game = None

    del black_agent.model
    del white_agent.model

    black_agent = white_agent = None

    import gc

    K.clear_session()
    gc.collect()

    if game_result.winner == Player.black:
        if black_agent is best_agent:
            return False
    else:
        if white_agent is best_agent:
            return False

    return True  # learner won this round


def evaluate_network(iteration, num_games, board_size, ratio=0.55, max_jobs=1):
    print(f'Evaluating iteration #{iteration}...')
    K.clear_session()

    experience = None
    submitted_count = 0
    learner_wins = 0
    decided = False

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        jobs = {}

        while submitted_count < num_games or len(jobs.keys()) > 0:
            # if the learner has won at least 55% of the total ratio, there's no point in continuing and we can
            # exit early
            #
            # similar reasoning for current best agent: if it has won more than 45% of the games, it's no longer
            # possible for the learner to dethrone the best agent
            decided = (learner_wins >= ratio * num_games) or ((submitted_count - learner_wins) > (1. - ratio))

            # don't submit any more games when winner is decided
            while not decided and len(jobs) < max_jobs and submitted_count < num_games:
                job = executor.submit(simulate)
                jobs[job] = job
                submitted_count += 1

            for completed in concurrent.futures.as_completed(jobs):
                try:
                    learner_won = completed.result()

                    if learner_won:
                        learner_wins += 1
                        print("learner won a game")
                    else:
                        print("best agent won a game")

                    del jobs[completed]

                    break

                except KeyboardInterrupt:
                    executor.terminate()
                    executor.join()
                    sys.exit(-1)

    return learner_wins >= ratio * num_games, learner_wins / num_games # (learner won, learner win ratio)
