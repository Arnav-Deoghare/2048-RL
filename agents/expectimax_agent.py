from agents.strategic_heuristic_agent import StrategicHeuristicAgent
from game.logic import move
import numpy as np


class ExpectimaxAgent:
    """Expectimax search for 2048.

    Max nodes: the player picks the move with the highest value.
    Chance nodes: a random tile spawns (2 w.p. 0.9, 4 w.p. 0.1) in a uniformly
    random empty cell; the value is the probability-weighted average.
    Leaves are scored with StrategicHeuristicAgent.evaluate_board.
    """

    P2 = 0.9
    P4 = 0.1
    LOSS_VALUE = -1e6  # value of a board with no legal moves

    def __init__(self, actions, seed=None, depth=2, max_chance_cells=6, adaptive_depth=True):
        self.sha = StrategicHeuristicAgent(actions, seed)
        self.evaluate = self.sha.evaluate_board

        self.actions = actions
        self.seed = seed
        self.base_depth = depth
        self.max_chance_cells = max_chance_cells
        self.adaptive_depth = adaptive_depth

        self.rng = np.random.default_rng(seed)
        self._cache = {}

    # ---------- public API ----------

    def choose_action(self, grid):
        orig_grid = grid.copy()
        depth = self._pick_depth(orig_grid)
        self._cache = {}

        best_value = None
        best_action = None
        for action in self.actions:
            next_grid = self._apply(orig_grid, action)
            if next_grid is None:
                continue
            value = self._chance_node(next_grid, depth - 1)
            if best_value is None or value > best_value:
                best_value = value
                best_action = action

        # No legal move (game over): return something valid so callers don't crash.
        if best_action is None:
            return self.actions[0]
        return best_action

    # ---------- search ----------

    def _max_node(self, grid, depth):
        if depth <= 0:
            return self.evaluate(grid)

        key = (grid.tobytes(), depth, "max")
        if key in self._cache:
            return self._cache[key]

        best = None
        for action in self.actions:
            next_grid = self._apply(grid, action)
            if next_grid is None:
                continue
            value = self._chance_node(next_grid, depth - 1)
            if best is None or value > best:
                best = value

        if best is None:  # terminal: no legal moves
            best = self.LOSS_VALUE

        self._cache[key] = best
        return best

    def _chance_node(self, grid, depth):
        if depth < 0:
            return self.evaluate(grid)

        key = (grid.tobytes(), depth, "chance")
        if key in self._cache:
            return self._cache[key]

        empty_cells = np.argwhere(grid == 0)
        if len(empty_cells) == 0:
            value = self._max_node(grid, depth)
            self._cache[key] = value
            return value

        # Bound the branching factor by sampling empty cells.
        if len(empty_cells) > self.max_chance_cells:
            idx = self.rng.choice(len(empty_cells), size=self.max_chance_cells, replace=False)
            empty_cells = empty_cells[idx]

        total = 0.0
        for r, c in empty_cells:
            g2 = grid.copy()
            g2[r, c] = 2
            g4 = grid.copy()
            g4[r, c] = 4
            total += self.P2 * self._max_node(g2, depth)
            total += self.P4 * self._max_node(g4, depth)

        value = total / len(empty_cells)
        self._cache[key] = value
        return value

    # ---------- helpers ----------

    def _apply(self, grid, action):
        """Return the grid after `action`, or None if the move changes nothing."""
        next_grid, _ = move(grid.copy(), action)
        if np.array_equal(grid, next_grid):
            return None
        return next_grid

    def _pick_depth(self, grid):
        if not self.adaptive_depth:
            return self.base_depth
        empties = int(np.count_nonzero(grid == 0))
        if empties <= 3:
            return self.base_depth + 2
        if empties <= 6:
            return self.base_depth + 1
        return self.base_depth