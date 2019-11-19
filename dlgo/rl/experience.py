import numpy as np


class ExperienceCollector:
    def __init__(self):     # 9.16 and 12.1
        # These can span many episodes
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []        # Added in 12.1
        # These are reset at the end of every episode
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []     # Added in 12.1

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []     # Added in 12.1

    # Saves a single decision in the current episode and
    # the agent is responsive for encoding the state and action.
    def record_decision(self, state, action, estimated_value=0):    # 12.2
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)      # Added in 12.2

    # Spreads the final reward across every action in the game.
    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        # Calculates the advantage of each decision
        for i in range(num_states):
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        # Resets the per-episode buffer
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []     # Added in 12.2

    def to_buffer(self):    # TheExperienceCollector accumulates Pythong lists; this converts to NumPy arrays
        return ExperienceBuffer(
                states=np.array(self.states),
                actions=np.array(self.actions),
                rewards=np.array(self.rewards))


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages):       # 9.13
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages    # Added in 12.4

    def serialize(self, h5file):        # 9.14
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)     # Added in 12.4


def combine_experience(collectors):     # 12.4
    combined_states = np.concatenate(
        [np.array(c.states) for c in collectors])
    combined_actions = np.concatenate(
        [np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate(
        [np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([
        np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(       # 12.4
        combined_states,
        combined_actions,
        combined_rewards,
        combined_advantages)


def load_experience(h5file):    # 9.15
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        actions=np.array(h5file['experience']['actions']),
        rewards=np.array(h5file['experience']['rewards']),
        advantages=np.array(h5file['experience']['advantages']))
