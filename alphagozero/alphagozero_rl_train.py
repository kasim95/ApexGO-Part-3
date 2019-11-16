import os
import multiprocessing
from shutil import copyfile

import dlgo.zero as zero
from dlgo.networks.zero import zero_model
import h5py
from alphagozero.generate_games import generate_games
from alphagozero.alphago_evaluate import evaluate_network


def train_cycle(curr_iteration, board_size, num_games, rounds_per_move=10, c=2.0, batch_size=1024, lr=0.01,
                games_parallel=2, eval_games=40, eval_ratio=0.55, eval_parallel=2):

    # generate new games to train on
    experience = generate_games(curr_iteration, num_games, board_size, rounds_per_move, c,
                                max_jobs=games_parallel)

    # add this new experience to existing experience (if any)
    if os.path.exists('agz_experience.h5'):
        with h5py.File('agz_experience.h5', 'a') as existing:
            previous_experience = zero.load_experience(h5py.File('agz_experience.h5'))
            experience = zero.combine_buffers(board_size, [previous_experience, experience])

    # save experience before training, since I crash keras a lot and don't want to waste time
    with h5py.File('agz_experience.h5', 'a') as expfile:
        experience.serialize(expfile)

    # open training agent
    if os.path.exists('agz_bot_train.h5'):
        with h5py.File('agz_bot_train.h5') as bot_file:
            learning_agent = zero.load_zero_agent(bot_file)

    else:
        model = zero_model(board_size)
        learning_agent = zero.ZeroAgent(model, zero.ZeroEncoder(board_size), rounds_per_move=rounds_per_move, c=c)

    # train agent on experienced games
    learning_agent.train(experience, lr, batch_size)

    # save trained bot
    with h5py.File('agz_bot_train.h5', 'w') as second_best:
        learning_agent.serialize(second_best)

    # evaluate network by playing the new version against the current best version

    if os.path.exists('agz_bot.h5'):
        # compare learning_agent with best_agent to determine victor
        # todo: turn down c to avoid exploration?

        learner_victory, learner_ratio = evaluate_network(iteration, eval_games, board_size, eval_ratio, eval_parallel)

        if learner_victory:
            # new victor!
            # todo: save old victor?
            print(f"**** learner agent victory **** {(learner_ratio * 100):.2f}%")

            with h5py.File('agz_bot.h5', 'w') as new_best:  # new top bot!
                learning_agent.serialize(new_best)
        else:
            print(f"**** learner agent failed **** {(learner_ratio * 100):.2f}%")

    # keep a copy of best agent for this iteration around, in case something ruins the current version
    copyfile('agz_bot.h5', f'agz_bot_i{iteration}.h5')

    del learning_agent.model
    learning_agent = None


if __name__ == '__main__':
    multiprocessing.freeze_support()

    t_board_size = 19
    t_parallel_games = os.cpu_count() - 1
    t_eval_parallel = t_parallel_games - 1  # since we're holding a model in memory already
    t_num_games_generate = 15  # how many self-play games to generate per iteration
    t_rounds_per_move = 250
    t_c = 2.0
    t_lr = 0.01
    t_batch_size = 1024
    t_num_eval_games = 50
    t_eval_ratio = 0.55

    iteration = 1

    while True:
        train_cycle(iteration, t_board_size, t_num_games_generate, t_rounds_per_move, t_c, t_batch_size, t_lr,
                    t_parallel_games, t_num_eval_games, t_eval_ratio, t_eval_parallel)

        iteration += 1
