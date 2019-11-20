from keras.layers import BatchNormalization, Conv2D, Flatten, Dense, Activation, Add, Dropout
from keras.models import Model, Input
import dlgo.zero as zero

density = 96


def create_residual_block(skip_from):
    conv = skip_from

    conv = Conv2D(density, (3, 3), padding='same', data_format='channels_first')(conv)
    conv = BatchNormalization(axis=1)(conv)
    conv = Activation(activation='relu')(conv)
    conv = Dropout(rate=0.25)(conv)

    conv = Conv2D(density, (3, 3), padding='same', data_format='channels_first')(conv)
    conv = BatchNormalization(axis=1)(conv)
    conv = Activation(activation='relu')(conv)

    conv = Conv2D(density, (3, 3), padding='same', data_format='channels_first')(conv)
    conv = BatchNormalization(axis=1)(conv)

    # skip connection
    conv = Add()([conv, skip_from])
    conv = Dropout(rate=0.25)(conv)

    conv = Activation(activation='relu')(conv)

    return conv


def create_policy_head(neck, board_size):
    neck = Conv2D(2, (2, 2), padding='same', data_format='channels_first')(neck)
    neck = BatchNormalization(axis=1)(neck)
    neck = Activation(activation='relu')(neck)

    neck = Flatten(data_format='channels_first')(neck)

    # neck = Dense(board_size * board_size + 1)(neck)  # +1 includes padding as a move (idx 361)
    # head = Activation(activation='softmax', name='move_probs')(neck)

    # +1 includes padding as a move (idx 361)
    head = Dense(board_size * board_size + 1, activation='softmax', name='move_probs')(neck)

    return head


def create_value_head(neck, board_size):
    neck = Conv2D(1, (2, 2), padding='same', data_format='channels_first')(neck)
    neck = BatchNormalization(axis=1)(neck)
    neck = Activation(activation='relu')(neck)

    neck = Flatten(data_format='channels_first')(neck)

    neck = Dense(board_size * board_size, activation='relu')(neck)
    neck = Dropout(rate=0.25)(neck)
    neck = Dense(density, activation='relu')(neck)
    head = Dense(1, activation='tanh', name='value_predictor')(neck)

    return head


def apex_model(board_size):
    residual_layers = 3

    encoder = zero.ZeroEncoder(board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input

    pb = Conv2D(density, (3, 3), padding='same', data_format='channels_first')(pb)
    pb = BatchNormalization(axis=1)(pb)
    pb = Activation(activation='relu')(pb)

    for i in range(residual_layers):
        pb = create_residual_block(pb)
        conv = pb

    neck = pb

    policy_head = create_policy_head(neck, board_size)
    value_head = create_value_head(neck, board_size)

    return Model(inputs=[board_input], outputs=[policy_head, value_head])
