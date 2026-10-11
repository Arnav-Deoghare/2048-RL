from agents.agent import RandomAgent
from game.environment import Environment

env = Environment()
agent = RandomAgent(env.actions)

for _ in range(5):
    action = agent.choose_action()
    print(action)
    assert action in env.actions