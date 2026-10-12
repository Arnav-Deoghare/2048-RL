import random

class RandomAgent:
    def __init__(self, actions, seed=None):

        self.actions = actions
        self.rng = random.Random(seed)

    def choose_action(self,state):
        return self.rng.choice(self.actions)