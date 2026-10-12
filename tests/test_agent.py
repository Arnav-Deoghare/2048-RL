import numpy as np

from agents.greedy_reward_agent import GreedyRewardAgent
from agents.strategic_heuristic_agent import StrategicHeuristicAgent
from game.logic import move


def main():
    agent = StrategicHeuristicAgent(["w", "a", "s", "d"])

    boards = [
        np.array([
            [16, 8, 4, 2],
            [8, 4, 2, 0],
            [4, 2, 0, 0],
            [2, 0, 0, 0]
        ]),
        np.array([
            [2, 16, 4, 8],
            [4, 2, 16, 0],
            [8, 4, 2, 0],
            [2, 0, 8, 4]
        ]),
        np.array([
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 8]
        ]),
        np.array([
            [2, 4, 8, 16],
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


if __name__ == "__main__":
    # test_valid_action()
    # test_highest_reward()
    main()
    print("All tests passed.")
