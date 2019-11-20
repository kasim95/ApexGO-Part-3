import numpy as np
from keras.optimizers import SGD
from dlgo import kerasutil
from dlgo.agent import Agent
from dlgo.encoders import get_encoder_by_name
from dlgo.zero.encoder import ZeroEncoder  # borrow zero's encoder


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.


class ApexTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}

        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)

        self.children = {}

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]

        return branch.total_value / branch.visit_count if branch.visit_count > 0 else 0.0

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count

        return 0


class ApexAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        super().__init__()

        self.model = model
        self.encoder = encoder
        self.num_rounds = rounds_per_move
        self.c = c

        self.collector = None

    def simulate_move(self, game_state, move):
        assert self.collector is not None

        root_state_tensor = self.encoder.encode(game_state)

        visit_counts = np.zeros(self.encoder.num_moves())
        visit_counts = np.array([
            0 for _ in range(self.encoder.num_moves())
        ])

        visit_counts[self.encoder.encode_move(move)] = 1

        self.collector.record_decision(
            root_state_tensor, visit_counts)

    def select_move(self, game_state):
        root = self.create_node(game_state)

        # todo: multithread this
        # todo: multithreaded rewindable board

        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(
                new_state, parent=node)

            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(
                root_state_tensor, visit_counts)

        root_state_tensor = self.encoder.encode(game_state)
        visit_counts = np.array([
            root.visit_count(
                self.encoder.decode_move_index(idx))
            for idx in range(self.encoder.num_moves())
        ])

        return max(root.moves(), key=root.visit_count)

    def set_collector(self, collector):
        self.collector = collector

    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        priors = priors[0]

        # add Dirichlet noise to encourage exploration
        if parent is None:
            noise = np.random.dirichlet(0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise

        value = values[0][0]

        move_priors = {
            self.encoder.decode_move_index(idx): p
            for idx, p in enumerate(priors)
        }
        new_node = ApexTreeNode(game_state, value, move_priors, parent, move)

        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def select_branch(self, node):
        total_n = node.total_visit_count

        # see 14.2.1 for explanation
        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)

            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.moves(), key=score_branch)

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]
        model_input = experience.states

        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_examples, 1))

        action_target = experience.visit_counts / visit_sums
        value_target = experience.rewards

        self.model.compile(SGD(lr=learning_rate), loss=['categorical_crossentropy', 'mse'], metrics=['accuracy'])
        self.model.fit(model_input, [action_target, value_target], batch_size=batch_size, epochs=5)

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file.create_group('meta')

        h5file['encoder'].attrs['board_size'] = self.encoder.board_size

        h5file.create_group('model')    # Uses built in Keras features to persist the model and its weights

        kerasutil.save_model_to_hdf5_group(
            self.model, h5file['model'])


def load_apex_agent(h5file, num_rounds, c):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    board_size= h5file['encoder'].attrs['board_size']
    encoder = ZeroEncoder(board_size)

    return ApexAgent(model, encoder, num_rounds, c)
