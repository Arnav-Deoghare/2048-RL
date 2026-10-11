import random

class RandomAgent:
    def __init__(self,actions):
        self.actions = actions
        
    def choose_action(self):
        return random.choice(self.actions)