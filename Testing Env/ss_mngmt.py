# Gymnasium imports
import gymnasium as gym 
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

import pygame
import pygame.gfxdraw

import networkx as nx

# Import helpers
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

from collections import deque

# Import stable baselines
# Probably not needed --> test ray lib instead
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class SS_Mngmt_Env(Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Define the action and observation space
    def __init__(self, render_mode = None):

        # To implement
        # storage capacities
        # campaign size
        # safety stock

        self.I = 30 # Initial stock level

        # Demand
        self.expected_demand = 6 # Expected demand for production
        self.actual_demand = 6 # Actual demand for production

        # Costs
        self.c_fixed = 50 # Fixed costs for ordering
        self.c_variable = 1 # Variable costs for ordering per unit
        self.p = 1 # Cost for storing each item
        self.stockout_cost = 1000 # Cost of stockout

        self.episode_length = 29 # Length of the episode

        # Order delay and queue
        self.order_delay = 3
        self.order_queue = deque(maxlen = self.order_delay)

        self.delivered = 0  # Delivered stock
        
        # Define action space
        # Change the action space for different campaign sizes
        self.action_space = Discrete(10)
        
        # Define observation space (Stock level) --> to add expected demand, etc.
        self.observation_space = Box(low = np.array([0, 0]), 
                                     high = np.array([100, 10]),
                                     dtype = np.int64)

        # Define the initial state
        self.state = np.array([self.I, self.expected_demand])

        # History
        self.reward_history = []
        self.stock_history = []
        self.demand_history = []
        self.expected_demand_history = []
        self.order_history = []
        self.delivery_history = []

        # Empty dataframe for plotting the history and analysis
        self.history = pd.DataFrame(columns = ['Stock Level', 'Order', 'Demand'])

        self.render_mode = render_mode
        self.screen_initialized = False


    # Defining the step function
    def step(self, action):
        # Returns the next state, reward and whether the episode is done

        # To Do: Campaign Size
        # To Do: Demand can only be fullfilled if sufficient stock is available demand is then placed in the backlog queue

        # Demand for production --> random for now
        self.demand = self.state[1] + random.randint(-1, 1)

        # Subtract the demand from the stock
        self.state[0] = self.state[0] - self.demand

        # If there are enough steps passed since the order was placed, add the order to the stock
        if len(self.order_queue) == self.order_delay:
            self.delivered = self.order_queue.popleft()
            self.state[0] += self.delivered
        
        # Add the order to the queue
        self.order_queue.append(action)

        # Calculate the reward based on order costs
        if action > 0:
            reward = - self.c_fixed - self.c_variable * action
        else:
            reward = 0

        # Check if the stock level is negative
        if self.state[0] <= 0:
            # If the stock level is negative, the cost is the stockout cost
            reward -=  self.stockout_cost
        else:
            # If the stock level is positive, the cost is the stock level
            reward -= self.state[0]

        # Calculate the cost of the stock
        # Reward is negative since we want to minimize the cost
        # reward = float(reward - (self.state))

        # Check if the episode is done
        done = self.episode_length == 0

        # Decrease the episode length
        self.episode_length -= 1

        obs = np.array([self.state[0], self.state[1]])

        # Append the state to the history
        self.reward_history.append(reward)
        self.stock_history.append(self.state[0])
        self.order_history.append(action)
        self.demand_history.append(self.demand)
        self.expected_demand_history.append(self.state[1])
        self.delivery_history.append(self.delivered)

        # Check if episode is done
        if self.episode_length <= 0: 
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Check if the episode is truncated
        truncated = False

        return obs, float(reward), done, truncated, info

    def render(self):
        # Just check episode lenghth and only plot the last one when using matplotlib          
        if self.render_mode is not None:
            if self.render_mode == "human":
                self.render_human()
                # self.render_pygame()

    def render_human(self):

        # To Do
        # Actual vs. Expected Demand

        # Create a list of timestamps for the order history
        timestamps = np.arange(len(self.order_history))

        # Setting up the plot
        fig, ax = plt.subplots(3, 2, figsize=(12, 9)) # 1 row and 2 columns
        bar_width = 0.35
        
        # Plotting the stock level
        ax[0, 0].plot(self.stock_history, color = 'blue', label = 'Stock Level')
        ax[0, 0].set_ylabel('Stock Level')
        # ax[0, 0].set_xlabel('Time')
        ax[0, 0].set_title('Stock Level vs. Time')
        ax[0, 0].set_xticks(np.arange(0, 30, step=5))
        ax[0, 0].set_yticks(np.arange(0, 100, step=10))

        # Plotting the order, delivered and demand
        ax[1, 0].bar(timestamps - bar_width, self.order_history, bar_width, color='blue', alpha = 0.5, label='Order')
        ax[1, 0].bar(timestamps, self.delivery_history, bar_width, color='red', alpha = 0.5, label='Delivery')
        ax[1, 0].bar(timestamps + bar_width, self.demand_history, bar_width, color='green', alpha=0.5, label='Demand')
        ax[1, 0].set_ylabel('Order/Delivery/Demand')
        ax[1, 0].set_title('Order vs. Delivery vs. Demand')
        ax[1, 0].set_xticks(np.arange(0, 30, step=5))
        ax[1, 0].legend(['Order', 'Delivery', 'Demand'])

        # Plotting the reward per timestep
        ax[0, 1].plot(self.reward_history, color = 'purple', label = 'Reward')
        ax[0, 1].set_ylabel('Reward')
        # ax[0, 1].set_xlabel('Time')
        ax[0, 1].set_title('Reward vs. Time')
        ax[0, 1].set_xticks(np.arange(0, 30, step=5))

        # Plotting the accumulated reward
        ax[1, 1].plot(np.cumsum(self.reward_history), color = 'orange', label = 'Accumulated Reward')
        ax[1, 1].set_ylabel('Accumulated Reward')
        # ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_title('Accumulated Reward vs. Time')
        ax[1, 1].set_xticks(np.arange(0, 30, step=5))

        # Plotting the expected demand and actual demand
        ax[2, 0].plot(self.expected_demand_history, color = 'blue', label = 'Expected Demand')
        ax[2, 0].plot(self.demand_history, color = 'green', label = 'Actual Demand')
        ax[2, 0].set_ylabel('Demand')
        ax[2, 0].set_title('Expected vs. Actual Demand')
        ax[2, 0].set_xticks(np.arange(0, 30, step=5))

        # Show the plot if the episode is done
        if self.episode_length == 0:
            # plt.tight_layout()
            plt.show()
        
        return


    def reset(self, seed = None):
        # Reset the state of the environment back to an initial state

        super().reset(seed = seed) # Reset the seed
        if seed is not None:
            random.seed(seed)

        # Reset the episode length
        self.episode_length = 29

        # Reset the order queue
        self.order_queue.clear()

        # Define the initial state
        self.state = np.array([self.I, self.expected_demand])

        obs = np.array([self.state[0], self.expected_demand])

        # # Append history to the dataframe
        # self.history['Stock Level'] = self.stock_history
        # self.history['Order'] = self.stock_history
        # self.history['Demand'] = self.demand_history

        # Reset the history
        self.reward_history = []
        self.stock_history = []
        self.order_history = []
        self.demand_history = []
        self.expected_demand_history = []
        self.delivery_history = []

        # Placeholder for info
        info = {}

        return obs, info