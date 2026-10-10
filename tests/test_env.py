from RL_env import Environment
import numpy as np

env = Environment()

env.reset()

env.grid = np.array([
    [0, 2, 4, 0],
    [0, 2, 4, 0],
    [2, 4, 8, 16],
    [32, 64, 128, 256]
])

print("Before:")
print(env.grid)

next_state, reward, done = env.step("w")

print("After:")
print(next_state)

print("Reward:", reward)
print("Done:", done)
state = env.reset()
print("State:", env.reset())
print("Actual grid:\n", env.grid)
print(state)
print(state.shape)
print("_________________________")




env = Environment()

print(env.actions)
state = env.reset()
print(state.shape)