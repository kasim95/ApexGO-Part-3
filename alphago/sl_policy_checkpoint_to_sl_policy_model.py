import sys
sys.path.append('../')

from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model
import h5py


ROWS, COLS = 19, 19

encoder = AlphaGoEncoder()
input_shape = (encoder.num_planes, ROWS, COLS)
alphago_sl_policy = alphago_model(input_shape=input_shape, is_policy_net=True)

# change checkpoint_path to the sl_policy checkpoint file path
checkpoint_path = 'alphago_sl_policy_1.h5'
alphago_sl_policy.load_weights(checkpoint_path)

alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

# change model_save_path to the name of the trained model
model_save_path = 'alphago_sl_policy_e3_2k.h5'
with h5py.File(model_save_path, 'w') as sl_agent_out:
    alphago_sl_agent.serialize(sl_agent_out)
