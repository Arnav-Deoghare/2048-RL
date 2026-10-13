import numpy as np

from agents.strategic_heuristic_agent import StrategicHeuristicAgent


def main():
    agent = StrategicHeuristicAgent(["w", "a", "s", "d"])

    boards = [
        # Board 1: Ordered tiles, largest tile in a corner
        np.array([
            [16, 8, 4, 2],
            [8, 4, 2, 0],
            [4, 2, 0, 0],
            [2, 0, 0, 0]
        ]),

        # Board 2: Disorganized board
        np.array([
            [2, 16, 4, 8],
            [4, 2, 16, 0],
            [8, 4, 2, 0],
            [2, 0, 8, 4]
        ]),

        # Board 3: Largest tile is not in a corner
        np.array([
            [2, 128, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]),

        # Board 4: Simple row with empty cells
        np.array([
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
    ]

    for i, grid in enumerate(boards, start=1):
        action = agent.choose_action(grid)
        score = agent.evaluate_board(grid)

        print(f"\nBoard {i}:")
        print(grid)
        print(f"Score: {score}")
        print(f"Chosen action: {action}")


if __name__ == "__main__":
    main()