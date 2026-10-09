import numpy as np
from logic import initialise_grid, move, spawn_tile, is_game_over

actions = ["w", "a", "s", "d"]
class Environment:
    def __init__(self):
        self.grid = initialise_grid()
        self.reward = 0


    def reset(self):
        self.grid = initialise_grid()
        self.done = is_game_over(self.grid)
        self.reward = 0
        return self.grid

    def step(self, action):

        before = self.grid.copy()
        self.grid, self.reward = move(self.grid, action)

        if not np.array_equal(before, self.grid):
            spawn_tile(self.grid)

        self.done = is_game_over(self.grid)
        return self.grid, self.reward, self.done


