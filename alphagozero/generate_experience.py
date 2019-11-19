import argparse
from .generate_games import generate_games

if __name__ == '__main__':


parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=19)
    parser.add_argument('--rounds', '-r', type=int, default=800)
    parser.add_argument('--cfactor', '-c', type=float, default=2.0)
    parser.add_argument('--games-per-chunk', '-g', type=int, default=144, help='Number of games per experience chunk')
    parser.add_argument('--num-chunks', '-p', type=int, default=6, help='Number of chunks to generate')
    parser.add_argument('--')
    # Customize through command line arguments
    args = parser.parse_args()

