import numpy as np
from keras import backend as K
from keras.optimizers import SGD

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil

__all__ = [
    'PolicyAgent',
    'load_policy_agent',
    'policy_gradient_loss',
]


# Keeping this around so we can read existing agents. But from now on
# we'll use the built-in crossentropy loss.
def policy_gradient_loss(y_true, y_pred):
    clip_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -1 * y_true * K.log(clip_pred)
    return K.mean(K.sum(loss, axis=1))


def normalize(x):
    total = np.sum(x)
    return x / total


# 9.7
class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self._model = model      # A Keras Sequential model instance
        self._encoder = encoder  # Implements the Encoder interface
        self._collector = None
        self._temperature = 0.0

    def predict(self, game_state):
        encoded_state = self._encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self._model.predict(input_tensor)[0]

    def set_temperature(self, temperature):
        self._temperature = temperature

    def set_collector(self, collector):  # 9.17
        self._collector = collector      # Allows the self-play driver program to attach a collector to the agent

    def select_move(self, game_state):  # 9.12 and 9.17
        board_tensor = self._encoder.encode(game_state)
        x = np.array([board_tensor])            # The Keras Predict call makes batch predictions,
        num_moves = self._encoder.board_width * self._encoder.board_height

        # move_probs = self._model.predict(x)[0]           # so you wrap your single board in an array and

        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            move_probs = self._model.predict(x)[0]

        # move_probs = clip_probs(move_probs)     # pull out the first item the resulting array
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        # move_probs = move_probs.astype(dtype=np.float64)
        candidates = np.arange(num_moves)  # Creates an array containing the index of every point on the board

        # Samples from the points on the board according to the policy, creates a ranked list of points to try
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:  # Loops over each point, checks if it's valid, and picks the first valid one
            point = self._encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye(game_state.board, point, game_state.next_player)
            if is_valid and (not is_an_eye):
                if self._collector is not None:     # At the time it chooses a move, notifies the collector of the deci
                    self._collector.record_decision(
                        state=board_tensor,
                        action=point_idx
                    )
                return goboard.Move.play(point)

        return goboard.Move.pass_turn()     # If you fall through here, there are not reasonable moves left.

    def serialize(self, h5file):    # 9.9
        h5file.create_group('encoder')  # stores enough information to reconstruct the board encoder
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')    # Uses build in Keras features to persist the model and its weights
        kerasutil.save_model_to_hdf5_group(
            self._model, h5file['model'])

    @staticmethod
    def prepare_experience_data(experience, board_width, board_height):     # 10.5
        experience_size = experience.actions.shape[0]
        target_vectors = np.zeros((experience_size, board_width * board_height))
        for i in range(experience_size):
            action = experience.actions[i]
            reward = experience.rewards[i]
            target_vectors[i][action] = reward
        return target_vectors

    def train(self, experience, lr=0.0000001, clipnorm=1.0, batch_size=512):  # 10.6
        self._model.compile(
            loss='categorical_crossentropy',    # The compile method assigns an optimizer to the model;
            optimizer=SGD(lr=lr, clipnorm=clipnorm))    # in this case, the SGD (stochastic gradient descent)

        target_vectors = self.prepare_experience_data(
            experience, self._encoder.board_width, self._encoder.board_height)

        self._model.fit(
            experience.states, target_vectors,
            batch_size=batch_size,
            epochs=1)


def load_policy_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(
        h5file['model'])    # Uses built in Keras functions to load the model structure and weights
    encoder_name = h5file['encoder'].attrs['name']      # Recovers the board encoder
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    return PolicyAgent(model, encoder)      # Reconstructs the agent
