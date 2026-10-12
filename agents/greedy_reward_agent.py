from game.logic import move
import numpy as np
class GreedyRewardAgent:
    def __init__(self, actions, seed=None):

        self.actions = actions

    def choose_action(self, grid):
        orig_grid = grid.copy()
        best_action = None
        best_reward = None
        best_changes_grid = False

        for action in self.actions:
            next_grid, local_reward = move(orig_grid.copy(), action)
            changes_grid = not np.array_equal(orig_grid, next_grid)

            if (best_reward is None or local_reward > best_reward
                or (local_reward == best_reward
                    and changes_grid
                    and not best_changes_grid)):
                
                best_reward = local_reward
                best_action = action
                best_changes_grid = changes_grid

        return best_action