#!/usr/bin/python
from dlgo.gtp import GTPFrontend
from dlgo.rl.ac import load_ac_agent
from dlgo.agent import termination
import h5py

model_file = h5py.File("superbot.hdf5", "r")

agent = load_ac_agent(model_file)
strategy = termination.get("opponent_passes")
termination_agent = termination.TerminationAgent(agent, strategy)

frontend = GTPFrontend(termination_agent)
frontend.run()
