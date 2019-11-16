import numpy as np


class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []


class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards, board_size):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards
        self.board_size = board_size

    def serialize(self, h5file):
        if 'experience' not in h5file:
            h5file.create_group('experience')
            h5file['experience'].create_dataset('states', data=self.states,
                                                maxshape=(None, 11, self.board_size, self.board_size), chunks=True)
            h5file['experience'].create_dataset('visit_counts', data=self.visit_counts,
                                                maxshape=(None, self.board_size * self.board_size + 1), chunks=True)
            h5file['experience'].create_dataset('rewards', data=self.rewards,
                                                maxshape=(None,), chunks=True)
            h5file['experience']['board_size'] = self.board_size
        else:
            states = h5file['experience']['states']
            visit_counts = h5file['experience']['visit_counts']
            rewards = h5file['experience']['rewards']

            states.resize(states.shape[0] + self.states.shape[0], axis=0)
            visit_counts.resize(visit_counts.shape[0] + self.visit_counts.shape[0], axis=0)
            rewards.resize(rewards.shape[0] + self.rewards.shape[0], axis=0)

            states[-self.states.shape[0]:] = self.states
            visit_counts[-self.visit_counts.shape[0]:] = self.visit_counts
            rewards[-self.rewards.shape[0]:] = self.rewards


def combine_experience(collectors, board_size):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return ZeroExperienceBuffer(combined_states, combined_visit_counts, combined_rewards, board_size)


def combine_buffers(board_size, buffers):
    combined_states = np.concatenate([np.array(b.states) for b in buffers])
    combined_visit_counts = np.concatenate([np.array(b.visit_counts) for b in buffers])
    combined_rewards = np.concatenate([np.array(b.rewards) for b in buffers])

    return ZeroExperienceBuffer(combined_states, combined_visit_counts, combined_rewards, board_size)


def load_experience(h5file):
    experience = h5file['experience']

    states = experience['states']
    visit_counts = experience['visit_counts']
    rewards = experience['rewards']
    board_size = experience['board_size']

    return ZeroExperienceBuffer(
        states=np.array(states),
        visit_counts=np.array(visit_counts),
        rewards=np.array(rewards),
        board_size=board_size
    )
