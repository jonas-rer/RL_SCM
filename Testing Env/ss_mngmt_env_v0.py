
# Gymnasium imports
import gymnasium as gym 
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

# Import helpers
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class SS_Mngmt_Env(Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Define the action and observation space
    def __init__(self, render_mode = None):

        # Define action space
        self.action_space = Discrete(10)
        
        # Define observation space
        # The stock can reach up to 100%
        self.observation_space = Box(low=np.array([0]), 
                                     high=np.array([100]),
                                     dtype=np.int64)

        # Define the initial state
        self.state = 50 + random.randint(-10, 10)

        # Length of the episode
        self.episode_length = 30

        # History of the stock level
        self.history = []

        self.render_mode = render_mode


    # Defining the step function
    def step(self, action):
        # Returns the next state, reward and whether the episode is done

        # action is the amount of stock to order
        # state is the current stock level
        # reward is the cost of the stock

        # Demand for production --> random for now
        demand = 6 + random.randint(-5, 5)

        # Calculate the cost of the stock
        # Reward is negative since we want to minimize the cost
        reward = float(-(action + self.state))

        # Update the state
        self.state = self.state + action - demand

        # Check if the episode is done
        done = self.episode_length == 0

        # Decrease the episode length
        self.episode_length -= 1

        obs = np.array([self.state])

        # Append the state to the history
        self.history.append(self.state)

        # Set placeholder for info
        info = {}

        # Check if the episode is truncated
        truncated = False

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode is not None:
            if self.render_mode == "human":
                self.render_human()

    def render_human(self):
        # Render the environment
        # Provide a visual representation of the environment the stock level
        # Stock level in y axis and time in x axis

        plt.show(block=False)
        plt.clf()
        
        plt.plot(self.history)
        plt.ylabel('Stock Level')
        plt.xticks(np.arange(0, 30, step=5))
        plt.yticks(np.arange(0, 100, step=10))
        plt.xlabel('Time')
        plt.draw()
        plt.pause(0.01)
        
        return

    def reset(self, seed = None):
        # Reset the state of the environment back to an initial state

        super().reset(seed = seed) # Reset the seed
        if seed is not None:
            random.seed(seed)

        # Reset the state
        self.state = 50 + random.randint(-20, 20)

        # Reset the episode length
        self.episode_length = 30

        obs = np.array([self.state])

        # Reset the history
        self.history = []

        # Placeholder for info
        info = {}

        return obs, info
