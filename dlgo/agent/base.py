class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()
        # From NotImplementedError to NotImplementedError(), pg.56

    def diagnostics(self):
        return {}
