import numpy as np

from agents.greedy_reward_agent import GreedyRewardAgent
from game.logic import move


def test_valid_action():
    agent = GreedyRewardAgent(["w", "a", "s", "d"])
    grid = np.zeros((4, 4), dtype=int)

    assert agent.choose_action(grid) in agent.actions


def test_highest_reward():
    agent = GreedyRewardAgent(["w", "a", "s", "d"])
    grid = np.array([
        [2, 2, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    rewards = {
        action: move(grid.copy(), action)[1]
        for action in agent.actions
    }

    chosen = agent.choose_action(grid)
    assert rewards[chosen] == max(rewards.values())


if __name__ == "__main__":
    test_valid_action()
    test_highest_reward()
    print("All tests passed.")
