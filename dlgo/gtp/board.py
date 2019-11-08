from dlgo.gotypes import Point
from dlgo.goboard_fast import Move

__all__ = ['coords_to_gtp_position', 'gtp_position_to_coords']

COLS = 'ABCDEFGHJKLMNOPQRST'


# Convert (row, col) tuple to GTP board locations
def coords_to_gtp_position(move):
    point = move.point
    return COLS[point.col - 1] + str(point.row)


# Convert a GTP board location to a (row, col) tuple
def gtp_position_to_coords(gtp_position):
    col_str, row_str = gtp_position[0], gtp_position[1:]
    point = Point(int(row_str), COLS.find(col_str.upper()) + 1)
    return Move(point)
