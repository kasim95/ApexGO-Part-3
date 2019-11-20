"""Since we're learning from better-ranked players, we can pretend we played out all these games for a quick
boost to training. Playing above these players will require an additional reinforcement learning step"""
from __future__ import print_function
from __future__ import absolute_import
import os
import os.path
import tarfile
import gzip
import shutil
import multiprocessing
from os import sys

import h5py

from dlgo.gosgf import SgfGame
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.data.index_processor import KGSIndex
from dlgo.encoders.base import get_encoder_by_name
from dlgo.zero.experience import ZeroExperienceCollector
from .apexagent import ApexAgent
from dlgo.zero.encoder import ZeroEncoder
from dlgo.scoring import compute_game_result
from dlgo.zero import combine_experience


def worker(jobinfo):
    try:
        clazz, zip_file, data_file_name = jobinfo
        clazz().process_zip(zip_file, data_file_name)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GameProcessor:
    def __init__(self, data_directory='data'):
        self.data_dir = data_directory

    def process_games(self):
        index = KGSIndex(data_directory=self.data_dir)
        #index.download_files()

        self.map_to_workers()

    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)

        tar_file = zip_file_name[0:-3]
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()

        return tar_file

    def process_zip(self, zip_file_name, data_file_name):
        print("processing {}...".format(data_file_name))

        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        name_list = zip_file.getnames()

        # use two strategies to get more data out:
        #  1) pretend we're both white and black
        #  2) rotate and mirror every game game
        # todo: 2

        encoder = ZeroEncoder(19)

        players = {
            # not going to actually use the model, so these are quite lightweight
            Player.black: ApexAgent(None, encoder),
            Player.white: ApexAgent(None, encoder)
        }

        for _, player in players.items():
            player.set_collector(ZeroExperienceCollector())

        counter = 0
        filename = f'{self.data_dir}/apexe_{data_file_name}exp.h5'

        for game_name in name_list:

            if not game_name.endswith('.sgf'):
                print(f'Warning: {game_name} is not a valid sgf')
                continue

            sgf_content = zip_file.extractfile(game_name).read()
            sgf = SgfGame.from_string(sgf_content)

            for _, player in players.items():
                player.collector.begin_episode()

            # todo: what the hell is first move done?
            game_state, first_move_done = self.get_handicap(sgf)

            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None

                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)

                    else:
                        move = Move.pass_turn()

                    color = Player.black if color == 'b' else Player.white

                    game_state = game_state.apply_move(move)

                    players[color].simulate_move(game_state, move)
                    first_move_done = True

            game_result = compute_game_result(game_state)

            players[game_result.winner].collector.complete_episode(1)
            players[game_result.winner.other].collector.complete_episode(-1)

            counter += 1

            print(f'Finished {game_name} - {counter}/{len(name_list)}')

            if counter % 100 == 0:
                exp = combine_experience([players[Player.black].collector, players[Player.white].collector], 19)

                with h5py.File(filename, 'w') as expfile:
                    exp.serialize(expfile)

                print(f'Saved checkpoint {filename}')

        print(f'Finishing {filename}...')

        exp = combine_experience([players[Player.black].collector, players[Player.white].collector], 19)

        with h5py.File(filename, 'w') as expfile:
            exp.serialize(expfile)

    @staticmethod
    def get_handicap(sgf):
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)

        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)

        return game_state, first_move_done

    def _get_games(self):
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)

        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > 2015:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))

        # temp
        return available_games[:1]

    def map_to_workers(self):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in self._get_games():
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name
            if not os.path.isfile(self.data_dir + '/apexe_' + data_file_name + 'exp.h5'):
                zips_to_process.append((self.__class__, zip_name, data_file_name))

        cores = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(worker, zips_to_process)
        try:
            _ = p.get()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(-1)
