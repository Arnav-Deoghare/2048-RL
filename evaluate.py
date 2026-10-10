from agents.agent import RandomAgent
from RL_env import Environment


env = Environment()
agent = RandomAgent(env.actions)
n_episodes = 100000

rewards = []
moves = []
max_tile = []
for episode in range(n_episodes):

    state = env.reset()
    done = False

    total_reward = 0
    n_moves = 0

    while not done:
        action = agent.choose_action()
        state,reward,done = env.step(action)

        total_reward += reward
        n_moves += 1

    # print(f"Game over! Total reward: {total_reward}, Number of moves: {n_moves}")
    rewards.append(total_reward)
    moves.append(n_moves)
    max_tile.append(2**max(state))
print(f"Average reward over {n_episodes} episodes: {sum(rewards)/n_episodes}")
print(f"Maximum reward over {n_episodes} episodes: {max(rewards)}")
print(f"Minimum reward over {n_episodes} episodes: {min(rewards)}")
print(f"Average number of moves over {n_episodes} episodes: {sum(moves)/n_episodes}")
print(f"Maximum number of moves over {n_episodes} episodes: {max(moves)}")
print(f"Minimum number of moves over {n_episodes} episodes: {min(moves)}")
print(f"Average maximum tile over {n_episodes} episodes: {sum(max_tile)/n_episodes}")
print(f"Maximum tile over {n_episodes} episodes: {max(max_tile)}")
print(f"Minimum tile over {n_episodes} episodes: {min(max_tile)}")