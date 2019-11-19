from __future__ import print_function
import subprocess
import re
import h5py

from dlgo.agent.termination import PassWhenOpponentPasses, TerminationAgent
from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Player
from dlgo.gtp.board import gtp_position_to_coords, coords_to_gtp_position
from dlgo.gtp.gtp_utils import SGFWriter
from dlgo.scoring import compute_game_result
from dlgo.utils import print_board
from dlgo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from dlgo.rl import load_value_agent


class LocalGtpBot:
    def __init__(self, go_bot, termination=None, handicap=0, opponent='gnugo', output_sgf='out.sgf', our_color='b'):
        self.bot = TerminationAgent(go_bot, termination)
        self.handicap = handicap
        self.game_state = GameState.new_game(19)
        self.sgf = SGFWriter(output_sgf)
        self.our_color = Player.black if our_color == 'b' else Player.white
        self.their_color = self.our_color.other

        cmd = self.opponent_cmd(opponent)
        pipe = subprocess.PIPE

        self.gtp_stream = subprocess.Popen(cmd, stdin=pipe, stdout=pipe, bufsize=1, universal_newlines=True)

        # state
        self._stopped = False

    @staticmethod
    def opponent_cmd(opponent):
        if opponent == 'gnugo':
            return ['../opponent_engines/gnugo-3.8/gnugo', '--mode', 'gtp']
        elif opponent == 'pachi':
            return ['../opponent_engines/Pachi/pachi']
        else:
            raise ValueError('Unknown bot name \'{}\''.format(opponent))

    def send_command(self, cmd):
        self.gtp_stream.stdin.write(cmd)

    def get_response(self):
        succeeded = False
        result = ''

        while not succeeded:
            for line in iter(self.gtp_stream.stdout.readline, ""):
                if line[0] == '=':
                    succeeded = True
                    line = line.strip()
                    result = re.sub('^= ?', '', line)
                    break

        return result

    def command_and_response(self, cmd):
        self.send_command(cmd)

        return self.get_response()

    def run(self):
        self.command_and_response('boardsize 19\n')
        self.set_handicap()
        self.play()
        self.sgf.write_sgf()

    def set_handicap(self):
        if self.handicap == 0:
            self.command_and_response('komi 7.5\n')
            self.sgf.append('KM[7.5]\n')
        else:
            stones = self.command_and_response('fixed_handicap {}\n'.format(self.handicap))
            sgf_handicap = "HA[{}]AB".format(self.handicap)

            for pos in stones.split(' '):
                move = gtp_position_to_coords(pos)
                self.game_state = self.game_state.apply_move(move)
                sgf_handicap += '[' + self.sgf.coordinates(move) + ']'

            self.sgf.append(sgf_handicap + '\n')

    def play(self):
        while not self._stopped:
            if self.game_state.next_player == self.our_color:
                self.play_our_move()
            else:
                self.play_their_move()

            print(chr(27) + '[2J')  # clears board
            print_board(self.game_state.board)
            print('Estimated result: ')
            print(compute_game_result(self.game_state))

    def play_our_move(self):
        import time

        start = time.time()

        move = self.bot.select_move(self.game_state)
        self.game_state = self.game_state.apply_move(move)

        our_name = self.our_color.name
        our_letter = our_name[0].upper()
        sgf_move = ''

        if move.is_pass:
            self.command_and_response('play {} pass\n'.format(our_name))
        elif move.is_resign:
            self.command_and_response('play {} resign\n'.format(our_name))
        else:
            pos = coords_to_gtp_position(move)

            self.command_and_response('play {0} {1}\n'.format(our_name, pos))
            sgf_move = self.sgf.coordinates(move)

        self.sgf.append(';{0}[{1}]\n'.format(our_letter, sgf_move))

        print(f'Took {time.time() - start} s to make our move')

    def play_their_move(self):
        their_name = self.their_color.name
        their_letter = their_name[0].upper()

        pos = self.command_and_response('genmove {}\n'.format(their_name))

        if pos.lower() == 'resign':
            self.game_state = self.game_state.apply_move(Move.resign())
            self._stopped = True
        elif pos.lower() == 'pass':
            self.game_state = self.game_state.apply_move(Move.pass_turn())
            self.sgf.append(';{}[]\n'.format(their_letter))

            if self.game_state.last_move.is_pass:
                self._stopped = True

        else:
            move = gtp_position_to_coords(pos)

            self.game_state = self.game_state.apply_move(move)
            self.sgf.append(';{0}[{1}]\n'.format(their_letter, self.sgf.coordinates(move)))


if __name__ == "__main__":
    import sys
    sys.path.append('../../')

    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))

    # change path of bot u want to play against
    # bot = load_prediction_agent(h5py.File('../../alphago/alphago_sl_policy_e13_1k.h5', 'r'))

    fast_policy = load_prediction_agent(h5py.File('alphago_sl_policy_e20_2k.h5', 'r'))
    strong_policy = load_policy_agent(h5py.File('alphago_rl_policy_e20_2k.h5', 'r'))
    value = load_value_agent(h5py.File('alphago_value_e20_2k.h5', 'r'))

    alphago = AlphaGoMCTS(strong_policy, fast_policy, value, lambda_value=0.5, num_simulations=5, depth=3, rollout_limit=10)
    gnu_go = LocalGtpBot(go_bot=alphago, termination=PassWhenOpponentPasses(), handicap=0, opponent='gnugo', our_color='w')
    gnu_go.run()
