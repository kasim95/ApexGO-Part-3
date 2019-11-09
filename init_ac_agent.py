
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input

from dlgo import encoders


# 12.5
def main():
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


if __name__ == '__main__':
    main()
