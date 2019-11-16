import os
import sys
import concurrent.futures
import time

from keras import backend as K

import h5py
import dlgo.zero as zero
from dlgo.networks.zero import zero_model

from dlgo.goboard_fast import GameState
from dlgo.gotypes import Player
from dlgo import scoring


def generate_game(board_size, game_id_str, rounds_per_move=10, c=2.0):
    start = time.time()
    print(f'Generating {game_id_str}...')

    game = GameState.new_game(board_size)
    encoder = zero.ZeroEncoder(board_size)

    # load current best agent, if any
    # has to be able to pass through cPickle which is why we don't just reuse it

    if os.path.exists('agz_bot.h5'):

        with h5py.File('agz_bot.h5') as bot_file:
            black_agent = zero.load_zero_agent(bot_file)
            white_agent = zero.load_zero_agent(bot_file)

    else:
        print(f'WARN: using default model to generate {game_id_str}')

        model = zero_model(board_size)

        black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=rounds_per_move, c=c)
        white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=rounds_per_move, c=c)

    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()

    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    c1.begin_episode()
    c2.begin_episode()

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)

    if game_result.winner == Player.black:
        c1.complete_episode(1)
        c2.complete_episode(-1)
    else:
        c1.complete_episode(-1)
        c2.complete_episode(1)

    combined = zero.combine_experience([c1, c2], board_size)

    c1 = c2 = game_result = None
    model = encoder = None
    game = None

    del black_agent.model
    del white_agent.model

    black_agent = white_agent = None

    import gc

    K.clear_session()
    gc.collect()

    return combined, game_id_str, time.time() - start


def generate_games(iteration, num_games, board_size, rounds_per_move, c, max_jobs=2):
    print(f'Beginning iteration #{iteration}...')
    K.clear_session()

    experience = None
    submitted_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        jobs = {}

        while submitted_count < num_games or len(jobs.keys()) > 0:
            while len(jobs) < max_jobs and submitted_count < num_games:
                job = executor.submit(generate_game,
                                      board_size,
                                      f'Iteration #[{iteration}] / Game {submitted_count + 1} of {num_games}',
                                      rounds_per_move,
                                      c)
                jobs[job] = job
                submitted_count += 1

            for completed in concurrent.futures.as_completed(jobs):
                try:
                    combined_exp_this_game, game_id, elapsed = completed.result()

                    print(f'Absorbing experience from {game_id}...')

                    experience = zero.combine_buffers(board_size, [experience, combined_exp_this_game])\
                        if experience is not None else combined_exp_this_game

                    print(f'Game {game_id} completed in {elapsed:.2f} seconds')

                    del jobs[completed]

                    break

                except KeyboardInterrupt:
                    sys.exit(-1)

    # todo: limit size somehow? AGZ limited to last 500,000 games
    # maybe not needed, we're not likely to reach that anyways

    return experience
