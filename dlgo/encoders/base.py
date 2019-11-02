# 6.1 and 6.2
import importlib


class Encoder:

    # Supports logging or saving the name of the encoder your model is using
    def name(self):
        raise NotImplementedError

    # Turns a GO board into numerical data
    def encode(self, game_state):
        raise NotImplementedError

    # Turns a GO board into an integer index (-1, 0, 1)
    def encode_point(self, point):
        raise NotImplementedError

    # Turns an integer index back into a GO board point
    def decode_point_index(self, point):
        raise NotImplementedError

    # Number of points on the board (width x height)
    def num_points(self):
        raise NotImplementedError

    # Shape of the encoded board structure
    def shape(self):
        raise NotImplementedError


# Reference coder by name, create a board matrix with an integer,
# and create an instance for encoder
def get_encoder_by_name(name, board_size):
    if isinstance(board_size, int):
        board_size = (board_size, board_size)
    module = importlib.import_module('dlgo.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)
