from keras.layers import Conv2D, Flatten, Dense
from keras.models import Model, Input

import dlgo.zero as zero
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Player
from dlgo import scoring
from dlgo.rl import GameRecord


def simulate_game(board_size, black_player, black_collector, white_player, white_collector):
    moves = []
    game = goboard.GameState.new_game(board_size)

    black_player.set_collector(black_collector)
    white_player.set_collector(white_collector)

    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def run():
    board_size = 9
    encoder = zero.ZeroEncoder(board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input

    for i in range(4):
        pb = Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu')(pb)

    policy_conv = Conv2D(2, (1, 1), data_format='channels_first', activation='relu')(pb)

    policy_flat = Flatten()(policy_conv)

    policy_output = Dense(encoder.num_moves(), activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1, 1), data_format='channels_first', activation='relu')(pb)

    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])

    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)

    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()

    for i in range(5):
        simulate_game(board_size, black_agent, c1, white_agent, c2)

    exp = zero.combine_experience([c1, c2])
    black_agent.train(exp, 0.01, 2048)


if __name__ == '__main__':
    run()