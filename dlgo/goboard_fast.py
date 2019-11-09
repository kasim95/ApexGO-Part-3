import copy
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from dlgo import zobrist
from .utils import MoveAge

neighbor_tables = {}
corner_tables = {}


def init_neighbor_table(dim):
    global neighbor_tables

    rows, cols = dim
    new_table = {}

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            pt = Point(row=row, col=col)

            full_neighbors = pt.neighbors()

            new_table[pt] = [n for n in full_neighbors if 1 <= n.row <= rows and 1 <= n.col <= cols]

    neighbor_tables[dim] = new_table


def init_corner_table(dim):
    global corner_tables

    rows, cols = dim
    new_table = {}

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            pt = Point(row=row, col=col)

            full_corners = [
                Point(row=pt.row - 1, col=pt.col - 1),
                Point(row=pt.row - 1, col=pt.col + 1),
                Point(row=pt.row + 1, col=pt.col - 1),
                Point(row=pt.row + 1, col=pt.col + 1)
            ]

            new_table[pt] = [n for n in full_corners if 1 <= n.row <= rows and 1 <= n.col <= cols]

    corner_tables = new_table


class IllegalMoveError(Exception):
    pass


class GoString:
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def without_liberty(self, point):
        new_liberties = self.liberties - {point}
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point):
        new_liberties = self.liberties | {point}
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, go_string):
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones)

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties

    def __deepcopy__(self, memodict=None):
        return GoString(self.color, self.stones, copy.deepcopy(self.liberties))


class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD

        global neighbor_tables
        dim = (num_rows, num_cols)

        if dim not in neighbor_tables:
            init_neighbor_table(dim)

        if dim not in corner_tables:
            init_corner_table(dim)

        self.neighbor_table = neighbor_tables[dim]
        self.corner_table = corner_tables[dim]
        self.move_ages = MoveAge(self)

    def neighbors(self, point):
        return self.neighbor_table[point]

    def corners(self, point):
        return self.corner_table[point]

    def place_stone(self, player, point):
        assert self.is_on_grid(point)

        if self._grid.get(point) is not None:
            print("Illegal play on %s by %s" % (str(point), str(player)))
        assert self._grid.get(point) is None

        # examine adjacent points
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []

        self.move_ages.increment_all()
        self.move_ages.add(point)

        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)

            if neighbor_string is None:
                liberties.append(neighbor)
            else:
                color = adjacent_same_color if neighbor_string.color == player else adjacent_opposite_color

                if neighbor_string not in color:
                    color.append(neighbor_string)

        new_string = GoString(player, [point], liberties)

        # merge adjacent strings of same color
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)

        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        self._hash ^= zobrist.HASH_CODE[point, None]
        self._hash ^= zobrist.HASH_CODE[point, player]

        # reduce liberties of adjacent strings of opposite color
        # (if opposite color strings now have zero liberties, remove them)
        for other_color_string in adjacent_opposite_color:
            replacement = other_color_string.without_liberty(point)  # type: GoString

            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            else:
                self._remove_string(other_color_string)

    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string):
        for point in string.stones:
            self.move_ages.reset_age(point)

            # might have created liberties for other strings
            for neighbor in self.neighbor_table[point]:
                neighbor_string = self._grid.get(neighbor)

                if neighbor_string is None:
                    continue

                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))

            # remove this point (stone) from board
            self._grid[point] = None
            self._hash ^= zobrist.HASH_CODE[point, string.color]
            self._hash ^= zobrist.HASH_CODE[point, None]

    def is_self_capture(self, player, point):
        friendly_strings = []

        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)  # type: GoString

            if neighbor_string is None:
                return False  # not a capture since this point has a liberty
            elif neighbor_string.color == player:
                friendly_strings.append(neighbor_string)
            else:
                if neighbor_string.num_liberties == 1:
                    return False

        return all(neighbor.num_liberties == 1 for neighbor in friendly_strings)

    def will_capture(self, player, point):
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)

            if neighbor_string is None or neighbor_string.color == player:
                continue
            elif neighbor_string.num_liberties == 1:
                return True  # this move would capture

        return False

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get(self, point):
        # returns content of a point on the board
        # could be None or a Player
        string = self._grid.get(point)

        return string.color if string is not None else None

    def get_go_string(self, point):
        # returns entire string of stones at a point (if any)
        return self._grid.get(point) or None

    def __eq__(self, other):
        return isinstance(other, Board) \
               and self.num_rows == other.num_rows and \
               self.num_cols == other.num_cols \
               and self.zobrist_hash() == other.zobrist_hash()

    def __deepcopy__(self, memodict=None):
        copied = Board(self.num_rows, self.num_cols)
        copied._grid = copy.copy(self._grid)
        copied._hash = self._hash

        return copied

    def zobrist_hash(self):
        return self._hash


class Move:
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)

    def __hash__(self):
        return hash((self.is_play, self.is_pass, self.is_resign, self.point))

    def __eq__(self, other):
        return (self.is_play, self.is_pass, self.is_resign, self.point) == \
               (other.is_play, other.is_pass, other.is_resign, other.point)

    def __str__(self):
        if self.is_pass:
            return "pass"
        if self.is_resign:
            return "resign"
        return "(r %d, c %d)" % (self.point.row, self.point.col)


class GameState:
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())})

        self.last_move = move

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)

        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        if self.last_move is None:
            return False

        if self.last_move.is_resign:
            return True

        second_last_move = self.previous_state.last_move

        return (self.last_move.is_pass and second_last_move.is_pass) if second_last_move is not None else False

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)

        return new_string.num_liberties == 0

    @property
    def situation(self):
        return self.next_player, self.board

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())

        return next_situation in self.previous_states

    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True

        return self.board.get(move.point) is None and \
            not self.is_move_self_capture(self.next_player, move) and \
            not self.does_move_violate_ko(self.next_player, move)

    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self)
        return game_result.winner

    def legal_moves(self):
        if self.is_over():
            return []

        moves = []

        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))

                if self.is_valid_move(move):
                    moves.append(move)

        # always legal
        moves.append(Move.pass_turn())
        moves.append(Move.resign())

        return moves
