import numpy as np
from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move
# from dlgo import kerasutil
import operator

__all__ = [
    'AlphaGoNode',
    'AlphaGoMCTS',
]


class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0):
        self.parent = parent  # <1>
        self.children = {}  # <1>

        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability  # <2>
        self.u_value = probability  # <3>
        # <1> Tree nodes have one parent and potentially many children.
        # <2> A node is initialized with a prior probability.
        # <3> The utility function will be updated during search.

    def select_child(self):
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + child[1].u_value)

    def expand_children(self, moves, probabilities):
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(probability=prob)

    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)  # <1>

        self.visit_count += 1  # <2>

        self.q_value += leaf_value / self.visit_count  # <3>

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                * self.prior_value / (1 + self.visit_count)  # <4>

        # <1> We update parents first to ensure we traverse the tree top to bottom.
        # <2> Increment the visit count for this node.
        # <3> Add the specified leaf value to the Q-value, normalized by visit count.
        # <4> Update utility with current visit counts.


class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                 lambda_value=0.5, num_simulations=1000,
                 depth=50, rollout_limit=100):
        Agent.__init__(self)
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()

    def select_move(self, game_state):
        # From current state play out a number of simulations
        for simulation in range(self.num_simulations):
            current_state = game_state
            node = self.root

            # Play moves until the specified depth is reached.
            for depth in range(self.depth):
                # If the current node doesn't have any children...
                if not node.children:
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)  # <4>

                    # expand them with probabilities from the strong policy.
                    node.expand_children(moves, probabilities)

                # If there are children, we can select one and play the corresponding move.
                move, node = node.select_child()
                current_state = current_state.apply_move(move)

            # Compute output of value network and a rollout by the fast policy.
            value = self.value.predict(current_state)
            rollout = self.policy_rollout(current_state)

            # Determine the combined value function.
            weighted_value = (1 - self.lambda_value) * value + self.lambda_value * rollout

            # Update values for this node in the backup phase
            node.update_values(weighted_value)

        # Pick most visited child of the root as next move.
        move = max(self.root.children,
                   key=lambda z: self.root.children.get(z).visit_count)

        self.root = AlphaGoNode()
        # If the picked move is a child, set new root to this child node.
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        return move

    def policy_probabilities(self, game_state):
        encoder = self.policy._encoder
        outputs = self.policy.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs

    def policy_rollout(self, game_state):
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)
            encoder = self.rollout_policy.encoder
            valid_moves = [m for idx, m in enumerate(move_probabilities)
                           if Move(encoder.decode_point_index(idx)) in game_state.legal_moves()]
            max_index, max_value = max(enumerate(valid_moves), key=operator.itemgetter(1))
            max_point = encoder.decode_point_index(max_index)
            greedy_move = Move(max_point)
            if greedy_move in game_state.legal_moves():
                game_state = game_state.apply_move(greedy_move)

        next_player = game_state.next_player
        winner = game_state.winner()
        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0

    def serialize(self, h5file):
        raise IOError("AlphaGoMCTS agent cant be serialized, consider serializing the 3 underlying," +
                      " neural networks instead")
