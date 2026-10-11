import csv
import random
from pathlib import Path
from statistics import mean

import numpy as np

from agents.agent import RandomAgent
from game.environment import Environment


# Reproducibility for the current implementation:
# Python random controls RandomAgent's action choices.
# NumPy random controls initial boards and tile spawning.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_EPISODES = 1000
RESULTS_FILE = Path("results") / "random_agent_results.csv"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def main():
    env = Environment()
    agent = RandomAgent(env.actions)
    episode_results = []

    for episode in range(1, N_EPISODES + 1):
        env.reset()
        done = False
        total_reward = 0
        n_moves = 0
        moves_to_2048 = None

        while not done:
            action = agent.choose_action()
            _, reward, done = env.step(action)

            total_reward += reward
            n_moves += 1

            # Record the first move on which the 2048 tile is reached.
            if moves_to_2048 is None and int(env.grid.max()) >= 2048:
                moves_to_2048 = n_moves

        episode_results.append({
            "episode": episode,
            "total_reward": int(total_reward),
            "moves": int(n_moves),
            "max_tile": int(env.grid.max()),
            "reached_2048": moves_to_2048 is not None,
            "moves_to_2048": moves_to_2048 if moves_to_2048 is not None else "",
        })

    # Save per-episode results for later analysis.
    fieldnames = [
        "episode",
        "total_reward",
        "moves",
        "max_tile",
        "reached_2048",
        "moves_to_2048",
    ]
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_results)

    rewards = [row["total_reward"] for row in episode_results]
    max_tiles = [row["max_tile"] for row in episode_results]
    episode_lengths = [row["moves"] for row in episode_results]
    successful_episodes = [
        row for row in episode_results if row["reached_2048"]
    ]
    successful_moves = [row["moves_to_2048"] for row in successful_episodes]

    success_rate = len(successful_episodes) / N_EPISODES * 100
    avg_moves_to_2048 = mean(successful_moves) if successful_moves else None

    print("Agent: RandomAgent")
    print(f"Episodes: {N_EPISODES}")
    print(f"Seed: {SEED}")
    print(f"Per-episode results saved to: {RESULTS_FILE.resolve()}")
    print()
    print("Evaluation metrics:")
    print(f"1. Average Total Reward: {mean(rewards):.2f}")
    print(f"2. 2048 Success Rate: {success_rate:.2f}%")
    print(f"3. Average Maximum Tile: {mean(max_tiles):.2f}")
    if avg_moves_to_2048 is None:
        print("4. Average Moves to 2048: N/A (no successful episodes)")
    else:
        print(f"4. Average Moves to 2048: {avg_moves_to_2048:.2f}")
    print(f"5. Average Episode Length: {mean(episode_lengths):.2f}")


if __name__ == "__main__":
    main()
