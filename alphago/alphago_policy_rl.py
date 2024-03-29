# import sys
# sys.path.append('../')

from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl.simulate import experience_simulation
import h5py

# 13.2
encoder = AlphaGoEncoder()

# make agents
sl_agent = load_prediction_agent(h5py.File('alphago_sl_policy_e13_1k.h5'))
sl_opponent = load_prediction_agent(h5py.File('alphago_sl_policy_e13_1k.h5'))

alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
opponent = PolicyAgent(sl_opponent.model, encoder)

# simulation
num_games = 1000
experience = experience_simulation(num_games, alphago_rl_agent, opponent)

alphago_rl_agent.train(experience)

with h5py.File('alphago_rl_policy_e13_1k.h5', 'w') as rl_agent_out:
    alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_experience_e13_1k.h5', 'w') as exp_out:
    experience.serialize(exp_out)
