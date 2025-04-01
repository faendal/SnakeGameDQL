import torch
import random
import numpy as np
from Snake import Snake

class Agent:
    
    RW_EAT: float = 1000
    RW_LOSE: float = -1000
    RW_CRASH: float = -100
    RW_STEP: float = -1
    
    def __init__(self, device):
        self.device = device
        self.actions = ["up", "down", "left", "right"]
    
    def choose_action(self, state, epsilon):
        """
        Choose an action based on the current state and epsilon-greedy strategy.
        :param state: Current state of the agent.
        :param epsilon: Probability of choosing a random action.
        :return: Chosen action.
        """
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)