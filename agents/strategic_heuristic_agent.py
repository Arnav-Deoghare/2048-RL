'''
The implementation plan
We'll build it in this order:

1. Empty cells: count how much space remains.
2. Monotonicity: measure how consistently tile values increase or decrease across rows and columns.
3. Smoothness: measure differences between neighboring tile values.
4. Corner strategy: reward keeping the largest tile in a corner.
5. Combine the scores: choose initial weights and evaluate the agent against the previous two.
6. Analyze failures: inspect whether the agent gets stuck, breaks its structure, or overvalues one heuristic.'''

import numpy as np
from game.logic import move

class StrategicHeuristicAgent:
    def __init__(self, actions, seed=None):

        self.actions = actions


    def choose_action(self, grid):
        orig_grid = grid.copy()
        best_score = None
        best_action = None

        for action in self.actions:
            next_grid, move_score = move(orig_grid.copy(), action)
            grid_score = self.evaluate_board(next_grid)

            changes_grid = not np.array_equal(orig_grid, next_grid)
            if not changes_grid:
                continue

            if best_score is None or grid_score > best_score:
                best_score = grid_score
                best_action = action

        return best_action

    def evaluate_board(self, grid):

        w_e = 1.0
        w_m = 1.0
        w_s = 1.0
        w_c = 100.0
        
        empty_cells = self.count_empty(grid)
        mono_score = self.monotonicity(grid)
        smooth_score = self.smoothness(grid)

        corner_score = self.check_corners(grid)

        board_score = (w_e*empty_cells - w_m*mono_score - w_s*smooth_score + w_c*corner_score)
        return board_score

    def count_empty(self, grid):
        empty = grid.size - np.count_nonzero(grid)
        return empty

    def monotonicity(self, grid):
        score = 0
        grid = np.log2(grid,
            where=(grid > 0),
            out=np.zeros_like(grid, dtype=float))

        for row in grid:
            diff_row = np.diff(row)

            increasing_score = sum(diff_row[diff_row > 0])
            decreasing_score = abs(sum(diff_row[diff_row < 0]))

            row_penalty = min(increasing_score, decreasing_score)
            score += row_penalty

        for column in grid.T:
            diff_column = np.diff(column)

            increasing_score = sum(diff_column[diff_column > 0])
            decreasing_score = abs(sum(diff_column[diff_column < 0]))

            column_penalty = min(increasing_score, decreasing_score)
            score += column_penalty

        return score
    
    def smoothness(self, grid):
        score = 0
        grid = np.log2(grid, where=(grid > 0), out=np.zeros_like(grid, dtype=float))

        left = grid[:, :-1]
        right = grid[:, 1:]
        mask = (left!=0) &(right!=0)
        difference = np.abs(left-right) 
        score += np.sum(difference[mask])

        up = grid [:-1, : ]
        down = grid[1:,:]
        mask = (up!=0) & (down!=0)
        difference = np.abs(up-down) 
        score += np.sum(difference[mask])

        return score

    def check_corners(self, grid):
        largest_tile = np.max(grid)

        corners= [grid[0, 0]]

        if largest_tile in corners:
            return 1
        return 0