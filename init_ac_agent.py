# python init_ac_agent.py --board-size 19 agents/ac/ac_v1.hdf5

import argparse
import h5py

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input
import dlgo.networks
from dlgo import encoders
from dlgo import rl


# 12.5
def main():
    """
    board_input = Input(shape=encoder.shape(), name='board_input')

    # Add as many convolutional layers as you like
    conv1 = Conv2D(64, (3, 3),
                   padding='same',
                   activation='relu')(board_input)
    conv2 = Conv2D(64, (3, 3),
                   padding='same',
                   activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3),
                   padding='same',
                   activation='relu')(conv2)

    flat = Flatten()(conv3)
    # This example uses hidden layers of size 512.
    # Experiment to find the best size.
    # The three hidden layers don't need to be the same size
    processed_board = Dense(512)(flat)

    # This output yields the policy function
    policy_hidden_layer = Dense(512, activation='relu')(processed_board)
    policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)

    # This output yields the value function
    value_hidden_layer = Dense(512, activation='relu')(processed_board)
    value_output = Dense(1, activation='tanh')(value_hidden_layer)

    model = Model(inputs=board_input,
                  outputs=[policy_output, value_output])
    """
    # added from gh repo
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--network', default='large')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('output_file')
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name('sevenplane', args.board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')

    processed_board = board_input
    network = getattr(dlgo.networks, args.network)
    for layer in network.layers(encoder.shape()):
        processed_board = layer(processed_board)

    policy_hidden_layer = Dense(args.hidden_size, activation='relu')(processed_board)
    policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)

    value_hidden_layer = Dense(args.hidden_size, activation='relu')(processed_board)
    value_output = Dense(1, activation='tanh')(value_hidden_layer)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])

    new_agent = rl.ACAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)
    #


if __name__ == '__main__':
    main()
