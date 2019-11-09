import numpy as np

from keras.optimizers import SGD

from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil
from dlgo.agent import Agent
from dlgo.agent.helpers import is_point_an_eye

__all__ = [
    'ACAgent',
    'load_ac_agent',
]


class ACAgent(Agent):   # 12.6
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 1.0
        self.last_state_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        x = np.array([board_tensor])

        # Because this is a two-output model, predict returns a tuple containing two NumPy arrays
        actions, values = self.model.predict(x)

        # predict is a batch call that can process several boards at once,
        # so you must select the first element of the array to get the probability
        # distribution you want.
        move_probs = actions[0]

        # The values are represented as a one-dimensional vector,
        # so you must pull out the first element to get the value as a plain float
        estimated_value = values[0][0]

        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(
                game_state.board, point, game_state.next_player)
            if move_is_valid and (not fills_own_eye):
                if self.collector is not None:
                    # Include the estimated value in the experience buffer
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value)
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    # lr (learning rate) and batch_size are tuning parameters for the optimizer;
    # refer to chapter 10 for more discussion
    def train(self, experience, lr=0.1, batch_size=128):    # 12.7
        opt = SGD(lr=lr)
        self.model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[1.0, 0.5])    # 1.0 applies to policy output and 0.5 applies to value output

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))
        for i in range(n):
            # This is the same as the encoding scheme in chapter 10, but weighted by the advantage
            action = experience.actions[i]
            policy_target[i][action] = experience.advantages[i]
            # This is the same as the encoding scheme in chapter 11
            reward = experience.rewards[i]
            value_target[i] = reward

        self.model.fit(
            experience.states,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1)

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

    def diagnostics(self):
        return {'value': self.last_state_value}


def load_ac_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return ACAgent(model, encoder)
