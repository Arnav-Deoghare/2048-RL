import csv
from pathlib import Path
from statistics import mean

import numpy as np

from agents.Random_agent import RandomAgent
from agents.greedy_reward_agent import GreedyRewardAgent
from agents.strategic_heuristic_agent import StrategicHeuristicAgent
from game.environment import Environment


SEED = 42
AGENT = "strategic_heuristic"
N_EPISODES = 1000
RESULTS_FILE = Path("results") / f"{AGENT}_agent_results.csv"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

TILE_MILESTONES = (64, 128, 256, 512, 1024,2048)
def create_agent(name, actions):
    agents = {
        "random": RandomAgent,
        "greedy_reward": GreedyRewardAgent,
        "strategic_heuristic": StrategicHeuristicAgent,
    }

    if name not in agents:
        raise ValueError(f"Unknown agent '{name}'. Available agents: {list(agents.keys())}")

    return agents[name](actions)

def main():
    env = Environment(seed=SEED)
    agent = create_agent(AGENT, env.actions)
    episode_results = []

    for episode in range(1, N_EPISODES + 1):
        env.reset()
        done = False
        total_reward = 0
        n_moves = 0
        ineffective_moves = 0
        moves_to_tile = {tile: None for tile in (*TILE_MILESTONES, 2048)}

        while not done:
            if n_moves >= 10000:
                print(f"Episode {episode} exceeded 10000 moves")
                break
            before = env.grid.copy()

            action = agent.choose_action(env.grid)
            _, reward, done = env.step(action)

            if np.array_equal(before, env.grid):
                ineffective_moves += 1
            total_reward += reward
            n_moves += 1
            current_max_tile = int(env.grid.max())

            # Record the first move on which each milestone is reached.
            for tile in moves_to_tile:
                if moves_to_tile[tile] is None and current_max_tile >= tile:
                    moves_to_tile[tile] = n_moves
        print(f"Finished episode {episode}: {n_moves} moves")
        row = {
            "episode": episode,
            "total_reward": int(total_reward),
            "moves": int(n_moves),
            "ineffective_moves": int(ineffective_moves),
            "max_tile": int(env.grid.max()),
        }

        for tile in TILE_MILESTONES:
            row[f"reached_{tile}"] = moves_to_tile[tile] is not None
            row[f"moves_to_{tile}"] = (
                moves_to_tile[tile] if moves_to_tile[tile] is not None else ""
            )

        # Keep the original 2048 fields for continuity with previous result files.
        row["reached_2048"] = moves_to_tile[2048] is not None
        row["moves_to_2048"] = (
            moves_to_tile[2048] if moves_to_tile[2048] is not None else ""
        )
        episode_results.append(row)

    # Save per-episode results for later analysis.
    fieldnames = ["episode", "total_reward", "moves", "ineffective_moves", "max_tile"]
    for tile in TILE_MILESTONES:
        fieldnames.extend([f"reached_{tile}", f"moves_to_{tile}"])
    fieldnames.extend(["reached_2048", "moves_to_2048"])

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_results)

    rewards = [row["total_reward"] for row in episode_results]
    max_tiles = [row["max_tile"] for row in episode_results]
    episode_lengths = [row["moves"] for row in episode_results]

    print(f"Agent: {AGENT}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Seed: {SEED}")
    print(f"Per-episode results saved to: {RESULTS_FILE.resolve()}")
    print()
    print("Evaluation metrics:")
    print(f"1. Average Total Reward: {mean(rewards):.2f}")
    print(f"2. Average Maximum Tile: {mean(max_tiles):.2f}")

    print("3. Tile Reach Rates:")
    for tile in TILE_MILESTONES:
        count = sum(row["max_tile"] >= tile for row in episode_results)
        rate = count / N_EPISODES * 100
        print(f"   - Reached >= {tile}: {rate:.2f}% ({count}/{N_EPISODES} episodes)")

    print("4. Average Moves to Tile (successful episodes only):")
    for tile in (TILE_MILESTONES[2:]):  # 256, 512, and 2048
        successful_moves = [
            row[f"moves_to_{tile}"]
            for row in episode_results
            if row[f"moves_to_{tile}"] != ""
        ]
        if successful_moves:
            print(f"   - Moves to {tile}: {mean(successful_moves):.2f}")
        else:
            print(f"   - Moves to {tile}: N/A (no episodes reached this tile)")

    print(f"5. Average Episode Length: {mean(episode_lengths):.2f}")
    total_ineffective_moves = sum(
        row["ineffective_moves"] for row in episode_results
    )
    total_moves = sum(episode_lengths)

    ineffective_move_rate = total_ineffective_moves / total_moves * 100
    print(f"6. Ineffective Move Rate: {ineffective_move_rate:.2f}%")

if __name__ == "__main__":
    main()