
import numpy as np
from .logic import initialise_grid, move, spawn_tile, is_game_over


class Environment:
    actions = ["w", "a", "s", "d"]

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.grid = initialise_grid(self.rng)
        self.reward = 0

    def reset(self):
        self.grid = initialise_grid(self.rng)
        self.done = is_game_over(self.grid)
        self.reward = 0
        return self.state_normalized(self.grid)

    def step(self, action):
        before = self.grid.copy()
        self.grid, self.reward = move(self.grid, action)

        if not np.array_equal(before, self.grid):
            spawn_tile(self.grid, self.rng)

        self.done = is_game_over(self.grid)
        return self.state_normalized(self.grid), self.reward, self.done

    def state_normalized(self,grid):
        grid = grid.flatten(order='C').copy()
        for i in range(len(grid)):
            if grid[i] != 0:
                grid[i] = np.log2(grid[i])
        return grid
