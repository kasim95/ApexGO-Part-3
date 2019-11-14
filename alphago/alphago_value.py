import sys
sys.path.append('../')

from dlgo.networks.alphago import alphago_model
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl import ValueAgent, load_experience
import h5py

# 13.3
# initialize values
rows, cols = 19, 19
encoder = AlphaGoEncoder()
input_shape = (encoder.num_planes, rows, cols)

alphago_value_network = alphago_model(input_shape)
alphago_value = ValueAgent(alphago_value_network, encoder)

# Train Value Network
experience = load_experience(h5py.File('alphago_rl_experience.h5', 'r'))
alphago_value.train(experience)

# Save model to file
with h5py.File('alphago_value.h5', 'w') as value_agent_out:
    alphago_value.serialize(value_agent_out)
