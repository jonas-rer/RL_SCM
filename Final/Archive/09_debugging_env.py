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
from stable_baselines3.common.env_checker import check_env

from Final.Environment.Archive.env_v7 import *
from Functions.visualization_functions import *

# %%
# Configuration of the network
with open("./Config/network_config_v0.json") as file:
    network_config = file.read()

EP_LENGTH = 52

stockout_cost = 2000
order_cost = 40
item_cost = 10
stock_cost = 0.02
item_prize = 2.5

# stockout_cost = 2000
# order_cost = 10
# item_cost = 1
# stock_cost = 0.1
# item_prize = 2

env = SS_Mngmt_Env(
    network_config=network_config,
    EP_LENGTH=EP_LENGTH,
    render_mode="human",
    model_type="PPO",
    stockout_cost=stockout_cost,
    order_cost=order_cost,
    item_cost=item_cost,
    stock_cost=stock_cost * item_cost,
    item_prize=item_prize * item_cost,
    order_quantities=[0, 30, 70],
    demand_mean=10,
    demand_std=2,
    demand_noise=0,
    demand_noise_std=2,
)

check_env(env, warn=True)


# %%

model = PPO("MultiInputPolicy", env)

# Training in increments and evaluating
total_timesteps = 1000000
evaluation_interval = 100000
for i in range(total_timesteps // evaluation_interval):
    model.learn(total_timesteps=evaluation_interval)

    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation after {(i+1) * evaluation_interval} timesteps:")
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
# %%
