import h5py

from dlgo import agent
from dlgo import rl


def main():
    # PLACEHOLDER VARIABLES (CHANGE BEFORE TRAINING MODEL)
    learning_agent_filename = h5py.File("")
    experience_files = ["", ""]
    learning_rate = 0.0001
    clipnorm = 0.5
    batchsize = 1024
    updated_agent_filename = "something"
    #
    learning_agent = agent.load_policy_agent(h5py.File(learning_agent_filename))  # 10.7
    for exp_filename in experience_files:
        exp_buffer = rl.load_experience(h5py.File(exp_filename))
        learning_agent.train(exp_buffer, lr=learning_rate, clipnorm=clipnorm, batch_size=batchsize)
    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
