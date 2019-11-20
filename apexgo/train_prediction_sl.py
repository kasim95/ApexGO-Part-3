import sys
import os
import glob
import random
import time

#sys.path.append('../')

from dlgo.data.parallel_processor2 import GoDataProcessor
from dlgo.zero.encoder import ZeroEncoder
from dlgo.networks.apexnetwork import apex_model


from keras.callbacks import ModelCheckpoint
from apexgo.ogs_to_experience import GameProcessor
from apexgo.apexagent import load_apex_agent, ApexAgent
from dlgo.zero.encoder import ZeroEncoder
from dlgo.zero.experience import load_experience
import h5py


def find_start_index():
    agent_files = glob.glob('agent/apex_agent_*.hdf5')
    return len(agent_files)


def display_elapsed(start, msg):
    print(f'Elapsed time for {msg}: {time.time() - start:.2f}')


def main():
    # sl data
    #encoder = ZeroEncoder(19)
    #processor = GameProcessor()

    #processor.process_games()

    experience_files = glob.glob('data/apexe_*.h5')
    iteration = 1
    cycles = 100
    agent_index = find_start_index() + 1

    # todo: load from file
    agent = ApexAgent(apex_model(19), ZeroEncoder(19))

    while True:
        print(f'Beginning iteration {iteration}...')
        start = time.time()

        # select an experience file to train with
        expfile = random.choice(experience_files)

        with h5py.File(expfile, 'r') as f:
            experience = load_experience(f)
        display_elapsed(start, 'loading experience')
        start = time.time()

        agent.train(experience, 0.01, 512)
        display_elapsed(start, 'training agent')
        start = time.time()

        agent_path = f'data/apex_agent_{agent_index}.hdf5'
        experience = None  # allow GC

        with h5py.File(agent_path) as agent_file:
            agent.serialize(agent_file)

        display_elapsed(start, 'serializing agent')
        agent_index += 1

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
