# %%

# Imports
# Gymnasium imports
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# Import helpers
import numpy as np
import pandas as pd
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from collections import deque

# Import stable baselines
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from Environment.env_v7 import *
from Functions.visualization_functions import *

# %%
# Configuration of the network
with open("./Final/Config/network_config_v0.json") as file:
    network_config = file.read()

stockout_cost_range = [500, 1000, 1500]  # Example values for stockout penalty
stock_cost_range = [0.01, 0.02, 0.03, 0.05]  # Example values for holding cost per unit
order_cost_range = [20, 30, 40, 50]  # Example order cost
item_cost_range = [5, 10, 15, 20]  # Selling price per item
item_price_range = [1.5, 2, 2.5, 3]  # Selling price per item
demand_mean_range = [5, 10, 15]  # Mean demand
demand_std_range = [1, 2, 3]  # Demand standard deviation

# %%
num_samples = 20  # Number of random samples to test

# Store results
random_results = []

for _ in range(num_samples):
    # Randomly select values within each range
    stockout_cost = random.choice(stockout_cost_range)
    stock_cost = random.choice(stock_cost_range)
    order_cost = random.choice(order_cost_range)
    item_cost = random.choice(item_cost_range)
    item_price = random.choice(item_price_range)
    demand_mean = random.choice(demand_mean_range)
    demand_std = random.choice(demand_std_range)

    # Create environment with random parameters
    env = SS_Mngmt_Env(
        network_config=network_config,
        EP_LENGTH=52,
        render_mode="human",
        stockout_cost=stockout_cost,
        order_cost=order_cost,
        item_cost=item_cost,
        stock_cost=stock_cost * item_cost,
        item_prize=item_price * item_cost,
        order_quantities=[0, 30, 80],
        demand_mean=10,
        demand_std=3,
        demand_noise=0,
        demand_noise_std=2,
    )

    # Train the model for a shorter time initially
    model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: env]),
        learning_rate=0.0007,
        gamma=0.99,
        verbose=0,
    )
    model.learn(total_timesteps=10_000)  # Short initial training

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, DummyVecEnv([lambda: env]), n_eval_episodes=5, render=False
    )

    # Log the results
    random_results.append(
        {
            "stockout_cost": stockout_cost,
            "stock_cost": stock_cost,
            "order_cost": order_cost,
            "item_cost": item_cost,
            "item_price": item_price,
            "demand_mean": demand_mean,
            "demand_std": demand_std,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
    )

# Convert to DataFrame for easy viewing
random_results_df = pd.DataFrame(random_results)
random_results_df.sort_values(by="mean_reward", ascending=False, inplace=True)
print(random_results_df.head())
