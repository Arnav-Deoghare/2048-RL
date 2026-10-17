from strategic_heuristic_agent import StrategicHeuristicAgent
from game.logic import move
import numpy as np
class ExpectimaxAgent:
    def __init__(self, actions, seed=None):
        self.sha = StrategicHeuristicAgent(actions, seed)
        self.evaluate = self.sha.evaluate_board
    
        self.actions = actions
        self.seed = seed


    def choose_action(self, grid):
        orig_grid = grid.copy()
        best_score = None
        best_action = None
        for action in self.actions:
            next_grid, move_score = move(orig_grid.copy(), action)
            changes_grid = not np.array_equal(orig_grid, next_grid)
            if not changes_grid:
                continue
            
            grid_score = self.evaluate(next_grid)
            if best_score is None or grid_score > best_score:
                best_score = grid_score
                best_action = action

            empty_cells = np.argwhere(next_grid ==0)

            for cell in empty_cells:
                r,c = cell
                next_next_grid = next_grid.copy()
                
                next_next_grid[r, c] = 2

        return best_action