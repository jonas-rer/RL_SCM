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
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pprint import pprint

from collections import deque

# Import stable baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class SS_Mngmt_Env(Env):
    """
    Supply Chain Management Environment
    Environment for supply chain management with a single product and multiple nodes.
    The action space constists of the order quantities for each node.
    The observation space consists of the inventory levels and the planned and actual demand for each node.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Define the action and observation space
    def __init__(
        self,
        EP_LENGTH=52,
        network_config=None,
        render_mode=None,
        model_type=None,
        stockout_cost=500,  # Cost of stockout
        stock_out_max=9,  # Maximum number of stockouts
        order_cost=5,  # Cost of each order
        item_cost=0.1,  # Cost of each item
        stock_cost=0.5,  # Cost of stock per unit
        item_prize=20,  # Prize of each item
        order_quantities=[0, 15, 50],  # Order quantities for each node
        demand_mean=10,  # Mean demand
        demand_std=2,  # Standard deviation of demand
        demand_noise=0,  # Mean noise in demand
        demand_noise_std=2,  # Standard deviation of noise in demand
        demand_prob=0.4,  # Probability of having demand
        intermediate_reward=1000,  # Intermediate reward
        progressive_stock_cost=False,  # Progressive stock cost
        kaggle=False,  # Kaggle mode (True or False)
    ):
        """
        Initialize the environment
        EP_LENGTH: int - Total length of the episode
        network_config: str - JSON string with network configuration
        render_mode: str - Render mode for the environment
        model_type: str - Type of model (e.g., PPO, A2C)
        stockout_cost: float - Cost of stockout
        stock_out_max: int - Maximum number of stockouts
        order_cost: float - Cost of each order
        item_cost: float - Cost of each item
        stock_cost: float - Cost of stock per unit
        item_prize: float - Prize of each item
        order_quantities: list - Order quantities for each node
        demand_mean: float - Mean demand
        demand_std: float - Standard deviation of demand
        demand_noise: float - Mean noise in demand
        demand_noise_std: float - Standard deviation of noise in demand
        demand_prob: float - Probability of having demand
        progressive_stock_cost: bool - Progressive stock cost
        kaggle: bool - Kaggle mode (True or False)
        """

        self.EP_LENGTH = EP_LENGTH  # Total length
        self.episode_length = EP_LENGTH  # Current length of the episode
        self.intermediate_reward = intermediate_reward  # Intermediate reward

        self.total_reward = 0

        self.model_type = model_type

        # Set the data path
        now = datetime.now()
        self.data_path = (
            f'./Data/{now.strftime("%Y-%m-%d")}_environment_data_{self.model_type}.csv'
        )

        # Set the fieldnames for the CSV file
        self.fieldnames = [
            "Time",
            "Node",
            "Stock",
            "Action",
            "Demand",
            "Delivery",
            "Reward",
            "Total Reward",
            "Backlog",
        ]

        self.file_initialized = False

        # Seting up the network
        self.network_config = network_config
        self.graph = nx.DiGraph()
        self.setup_network(self.network_config)

        self.lead_times = nx.get_edge_attributes(self.graph, "L")

        # Number of nodes excluding 'S' and 'D'
        num_nodes = len(self.graph.nodes) - 2

        # Define the costs
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.item_cost = item_cost
        self.stock_cost = stock_cost
        self.stock_out_max = stock_out_max
        self.item_prize = item_prize
        self.progressive_stock_cost = progressive_stock_cost

        self.stock_out_counter = 0

        self.order_quantities = order_quantities

        # Order delay and queue
        self.order_queues = self.order_queue(initial_order=order_quantities[1])

        # Backlog queue for each node
        self.backlog_queues = self.backlog_queue()

        # Define action space
        n_actions = len(order_quantities)
        n_nodes = len(self.graph.nodes) - 2
        action_choices = np.full(n_nodes, n_actions)
        self.action_space = MultiDiscrete(action_choices)

        max_lead_time = max([data["L"] for _, _, data in self.graph.edges(data=True)])
        self.observation_space = Dict(
            {
                "inventory_levels": Box(
                    low=0, high=1000, shape=(num_nodes,), dtype=np.float32
                ),
                "current_demand": Box(
                    low=0, high=1000, shape=(num_nodes,), dtype=np.float32
                ),
                "backlog_levels": Box(
                    low=0, high=1000, shape=(num_nodes,), dtype=np.float32
                ),
                "order_queues": Box(
                    low=0, high=1000, shape=(num_nodes, max_lead_time), dtype=np.float32
                ),
                "lead_times": Box(
                    low=1, high=max_lead_time, shape=(num_nodes,), dtype=np.int32
                ),
            }
        )

        # Setting up the initial state
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.demand_noise = demand_noise
        self.demand_noise_std = demand_noise_std
        self.demand_prob = demand_prob

        self.planned_demands = self.planned_demand(
            self.demand_mean, self.demand_std, self.demand_prob
        )
        self.actual_demands = self.actual_demand(
            self.planned_demands, self.demand_noise_std, self.demand_noise
        )

        # Collect initial inventories from the graph
        initial_inventories = []
        for node in self.graph.nodes:
            if node not in ["S", "D"]:
                initial_inventories.append(self.graph.nodes[node].get("I", 0))

        initial_inventories = np.array(initial_inventories, dtype=np.float32).flatten()

        self.state = {
            "inventory_levels": initial_inventories.astype(np.float32),
            "planned_demand": self.planned_demands,
            "actual_demand": self.actual_demands,
            "current_demand": self.actual_demands[0],
            "backlog_levels": np.zeros(num_nodes),
            "order_queue_status": np.zeros(num_nodes),
        }

        # Prep to save the data
        self.inventory = initial_inventories
        self.stock_history = [self.inventory.tolist()]
        self.reward_history = [np.sum(initial_inventories * self.stock_cost * -1)]

        # Kaggle mode
        self.kaggle = kaggle

        # Render mode
        self.render_mode = render_mode
        self.screen_initialized = False

    # Defining the step function
    def step(self, action):
        """
        Executes one step in the environment.
        Starts by processing the orders and updating the inventory levels for each node.
        Then, it computes the reward based on the order costs and stock level.
        Finally, it checks if the episode is done and returns the next state, reward, and whether the episode is done.
        """

        # Returns the next state, reward and whether the episode is done
        timestep = self.EP_LENGTH - self.episode_length

        # num_nodes = len(self.graph.nodes) - 2

        # Retrieve the current inventory levels
        self.inventory = self.state["inventory_levels"]
        inventory_levels = np.copy(self.inventory)
        reward = 0

        # Retrieve the actual demand for the current timestep
        self.current_demand = self.actual_demands[timestep].astype(np.float32)

        # Add every first element of the order queues to the history
        self.new_order = [self.order_quantities[i] for i in action]

        # For visualization and history data
        self.orders = np.array(
            [
                self.order_queues[node][0]
                for node in self.graph.nodes
                if node not in ["S", "D"]
            ]
        )

        # Process the orders and update the inventory levels for each node
        for node in self.graph.nodes:
            if node not in ["S", "D"]:
                node_index = self.node_to_index(node)

                # Deduct costs for placing new orders
                if self.new_order[node_index] > 0:
                    reward -= self.order_cost + (
                        self.new_order[node_index] * self.item_cost
                    )

                # Fulfill orders from the queue
                order = self.order_queues[node].popleft()
                inventory_levels[node_index] += order

                # Attempt to meet current demand
                node_demand = self.current_demand[node_index]
                if inventory_levels[node_index] >= node_demand:
                    # Enough stock to meet demand
                    inventory_levels[node_index] -= node_demand
                    reward += node_demand * self.item_prize
                else:
                    # Insufficient stock - add unmet demand to backlog and apply penalty
                    unmet_demand = node_demand - inventory_levels[node_index]
                    inventory_levels[node_index] -= node_demand - unmet_demand
                    reward += (node_demand - unmet_demand) * self.item_prize
                    reward -= self.stockout_cost * unmet_demand  # Apply stockout cost
                    self.backlog_queues[node].append(unmet_demand)

                    # Increment the stockout counter
                    self.stock_out_counter += 1

                # Process backlog with any remaining stock
                while self.backlog_queues[node] and inventory_levels[node_index] > 0:
                    backlog_demand = self.backlog_queues[node][0]
                    if inventory_levels[node_index] >= backlog_demand:
                        inventory_levels[node_index] -= backlog_demand
                        reward += backlog_demand * self.item_prize
                        self.backlog_queues[node].popleft()
                    else:
                        break  # Not enough stock to clear the backlog completely

                # backlog penalty
                reward -= self.stockout_cost * len(self.backlog_queues[node])

                # Replenish order queue
                self.order_queues[node].append(self.new_order[node_index])

        if self.progressive_stock_cost == False:
            # Compute the reward based on the order costs and stock level
            reward -= np.sum(inventory_levels) * self.stock_cost
        elif self.progressive_stock_cost == True:
            reward -= np.sum(
                [
                    self.quadratic_stock_cost(self.stock_cost, inv)
                    for inv in inventory_levels
                ]
            )

        # Penalty if the episode cannot be completed
        if self.stock_out_counter >= self.stock_out_max:
            reward -= (
                self.episode_length * self.stockout_cost * (len(self.graph.nodes) - 2)
            )

        # Intermediate reward
        if timestep % 2 == 0:
            reward += self.intermediate_reward

        # Update the reward
        self.total_reward += reward

        # Decrease the episode length
        self.episode_length -= 1

        inventory_levels = inventory_levels.flatten()
        self.inventory = inventory_levels

        self.state = {
            "inventory_levels": inventory_levels.astype(np.float32),
            "planned_demand": self.planned_demands,
            "actual_demand": self.actual_demands,
            "current_demand": self.actual_demands[timestep],
            "backlog_levels": self.backlog_queues,
            "order_queue_status": self.order_queues,
        }

        max_lead_time = max([data["L"] for _, _, data in self.graph.edges(data=True)])
        obs = {
            "inventory_levels": self.inventory.astype(np.float32),
            "current_demand": self.actual_demands[timestep].astype(np.float32),
            "backlog_levels": np.array(
                [len(queue) for queue in self.backlog_queues.values()], dtype=np.float32
            ),
            "order_queues": np.array(
                [
                    list(self.order_queues[node])
                    + [0] * (max_lead_time - len(self.order_queues[node]))
                    for node in self.graph.nodes
                    if node not in ["S", "D"]
                ],
                dtype=np.float32,
            ),
            "lead_times": np.array(
                [
                    len(self.order_queues[node])
                    for node in self.graph.nodes
                    if node not in ["S", "D"]
                ],
                dtype=np.int32,
            ),
        }

        # Update the history data
        self.reward_history.append(reward)
        self.stock_history.append(list(self.inventory))

        self.log_step_data(timestep, action, reward)

        # Check if the episode is done
        done = self.episode_length == 0

        # Check if episode is done
        if self.stock_out_counter >= self.stock_out_max:

            done = True

        elif self.episode_length <= 0:

            done = True

        else:

            done = False

        # Set placeholder for info
        info = {}

        # Check if the episode is truncated
        truncated = False

        return obs, float(reward), done, truncated, info

    def quadratic_stock_cost(self, stock_cost, inventory_level):
        """
        Quadratic stock cost function.
        """
        return stock_cost * (inventory_level**2)  # Quadratic cost

    def log_step_data(self, timestep, action, reward):

        if not self.file_initialized:
            with open(self.data_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            self.file_initialized = True  # Mark as initialized

        for n in range(len(self.inventory)):
            node_name = self.get_node_name(n)
            row = {
                "Time": timestep + 1,
                "Node": node_name,
                "Stock": self.inventory[n],
                "Action": self.new_order[n],
                "Demand": self.current_demand[n],
                "Delivery": self.orders[n],
                "Reward": reward,
                "Total Reward": self.total_reward,
                "Backlog": len(self.backlog_queues[node_name]) > 0,
            }

            # Append the row to the CSV file
            with open(self.data_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(row)

    def reward_function(self):
        # TODO - Implement a custom reward function

        return 0

    def render(self):
        # Just check episode lenghth and only plot the last one when using matplotlib
        if self.render_mode is not None:
            if self.render_mode == "human":
                self.render_human()

    def render_human(self):
        """
        Renders the environment in human mode.
        Useful for debugging and visualization.
        """

        print("*" * 50)
        print("\nEpisode Information")
        print(f"Episode Length: {self.EP_LENGTH - self.episode_length}")
        if len(self.stock_history) > 1:
            print(f"Stock Level (Previous Timestep): {self.stock_history[-2]}")
        else:
            print(
                "Stock Level (Previous Timestep): No previous timestep data available"
            )
        print(f"Stock Level: {self.inventory}")
        print(
            f"Planned Demand: {self.planned_demands[self.EP_LENGTH - self.episode_length - 1]}"
        )
        print(f"Actual Demand: {self.current_demand}")
        print(f"Action: {self.new_order}")
        print(f"Deliveries: {self.orders}")
        # print(
        #     f"Previous Reward: {self.reward_history[self.EP_LENGTH - self.episode_length - 1]}"
        # )
        print(f"Step Reward: {self.reward_history[-1]}")
        print(f"Total Reward: {self.total_reward}")

        print("\nBacklog:")
        print([len(queue) > 0 for queue in self.backlog_queues.values()])
        # pprint(self.backlog_queues, indent=4)

        print("\nOrder Queue:")
        pprint(self.order_queues, indent=4)
        print()

        # print("Stockout Cost: ", self.stockout_cost)
        print("\nStockout Counter: ", self.stock_out_counter)

        return

    def setup_network(self, network_config=None):
        """
        Sets up the network graph based on the configuration provided.
        """
        config = json.loads(network_config)

        # Add nodes to the graph
        for node, attributes in config["nodes"].items():
            self.graph.add_node(node, **attributes)

        # Add edges to the graph with lead times
        for edge in config["edges"]:
            self.graph.add_edge(edge["source"], edge["target"], L=edge["L"])

    def render_network(self):
        """
        Renders the network graph using NetworkX and Matplotlib.
        """

        print("Node Attributes:")
        for node, attributes in self.graph.nodes(data=True):
            print(f"Node {node}: {attributes}")

        pos = graphviz_layout(self.graph, prog="dot")

        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color="lightblue")
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=self.graph.edges(), arrowstyle="->", arrowsize=20
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_family="sans-serif")

        edge_labels = nx.get_edge_attributes(self.graph, "L")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Supply Chain Network Graph", fontsize=15)

        # Display the plot
        plt.axis("off")
        plt.show()

    def node_to_index(self, node):
        """
        Returns the index of the node given its name.
        """
        return list(self.graph.nodes).index(node)

    def get_node_name(self, index):
        """
        Returns the name of the node given its index.
        """
        return list(self.graph.nodes)[index]

    def planned_demand(self, demand_mean=10, demand_std=2, demand_prob=0.8):
        """
        Generates planned demand for each edge in the network over the whole episode.
        """

        edges_leading_to_D = [edge for edge in self.graph.edges if edge[1] == "D"]

        planned_demand = np.zeros((self.EP_LENGTH, len(edges_leading_to_D)))

        for i, edge in enumerate(edges_leading_to_D):
            for j in range(self.EP_LENGTH):
                # Introduce a probability of having demand
                if np.random.rand() < demand_prob:
                    planned_demand[j, i] = int(
                        np.random.normal(demand_mean, demand_std)
                    )

        return planned_demand

    def planned_demand_even(self, demand_mean, demand_std):
        """
        Generates planned demand for each edge in the network over the whole episode.
        The demand is distributed evenly, occurring only at fixed intervals (e.g., every fifth timestep).
        """

        # Get edges leading to "D"
        edges_leading_to_D = [edge for edge in self.graph.edges if edge[1] == "D"]

        # Initialize demand array with zeros
        planned_demand = np.zeros((self.EP_LENGTH, len(edges_leading_to_D)))

        for i, edge in enumerate(edges_leading_to_D):
            # Determine timesteps where demand occurs (e.g., every fifth timestep)
            timesteps_with_demand = np.arange(0, self.EP_LENGTH, 5)

            for j in timesteps_with_demand:
                # Generate demand from a normal distribution
                demand = max(
                    0, np.random.normal(demand_mean, demand_std)
                )  # Ensure non-negative demand
                planned_demand[j, i] = int(demand)

        return planned_demand

    def actual_demand(self, planned_demand, demand_noise_std, demand_noise):
        """
        Generates a random actual demand for each edge in the network based on the planned demand from the current timestep.
        """

        actual_demand = np.copy(planned_demand)

        for i in range(actual_demand.shape[0]):
            for j in range(actual_demand.shape[1]):
                # Add a small random noise to the planned demand
                if planned_demand[i, j] > 0:
                    noise = np.random.normal(demand_noise, demand_noise_std)
                    # Ensure actual demand is not less than 0
                    actual_demand[i, j] = int(max(0, actual_demand[i, j] + noise))

        return actual_demand

    def actual_demand_extremes(self, planned_demand, demand_noise_std, demand_noise):
        """
        Generates a random actual demand for each edge in the network based on the planned demand from the current timestep.
        The to create a more interesting scenario the demand is multiplied by a random factor every now and then.
        """

        actual_demand = np.copy(planned_demand)

        for i in range(actual_demand.shape[0]):
            for j in range(actual_demand.shape[1]):
                # Add a small random noise to the planned demand
                if planned_demand[i, j] > 0:
                    noise = np.random.normal(demand_noise, demand_noise_std)
                    # Ensure actual demand is not less than 0
                    actual_demand[i, j] = int(max(0, actual_demand[i, j] + noise))

        return actual_demand

    def order_queue(self, initial_order=10):
        """
        Creates a dictionary for the order queues for each node in the network.
        """

        order_queues = {}

        for node in self.graph.nodes:

            if node not in ["S", "D"]:
                in_edges = list(self.graph.in_edges(node, data=True))

                if in_edges:
                    lead_time = in_edges[0][2]["L"]
                    order_queues[node] = deque(
                        [initial_order] + [0] * (lead_time - 1), maxlen=lead_time
                    )

        return order_queues

    def backlog_queue(self):
        """
        Creates a dictionary for the backlog queues for each node in the network.
        """

        backlog_queues = {}

        for node in self.graph.nodes:
            if node not in ["S", "D"]:

                in_edges = list(self.graph.in_edges(node, data=True))
                if in_edges:
                    backlog_queues[node] = deque()

        return backlog_queues

    def save_state(self):
        """
        Saves the current state of the environment.
        Used for greedy algorithm.
        """

        return {
            "episode_length": self.episode_length,
            "inventory": np.copy(self.inventory),
            "total_reward": self.total_reward,
            "state": self.state,
            "order_queues": {k: deque(v) for k, v in self.order_queues.items()},
            "backlog_queues": {k: deque(v) for k, v in self.backlog_queues.items()},
        }

    def load_state(self, saved_state):
        """
        Loads the state of the environment.
        Used for greedy algorithm.
        """

        self.episode_length = saved_state["episode_length"]
        self.inventory = saved_state["inventory"]
        self.total_reward = saved_state["total_reward"]
        self.state = saved_state["state"]
        self.order_queues = {
            k: deque(v) for k, v in saved_state["order_queues"].items()
        }
        self.backlog_queues = {
            k: deque(v) for k, v in saved_state["backlog_queues"].items()
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        super().reset(seed=seed)  # Reset the seed
        if seed is not None:
            random.seed(seed)

        # Reset the episode length
        self.episode_length = self.EP_LENGTH

        self.file_initialized = False

        self.total_reward = 0

        # Reset the network
        self.graph = nx.DiGraph()
        self.setup_network(self.network_config)

        num_nodes = len(self.graph.nodes) - 2

        # Order delay and backlog queue
        self.order_queues = self.order_queue(initial_order=self.order_quantities[1])
        self.backlog_queues = self.backlog_queue()

        self.stock_out_counter = 0

        # Define the initial state
        self.planned_demands = self.planned_demand(
            self.demand_mean, self.demand_std, self.demand_prob
        ).astype(np.float32)

        self.actual_demands = self.actual_demand(
            self.planned_demands, self.demand_noise_std, self.demand_noise
        ).astype(np.float32)

        self.current_demand = self.actual_demands[0].astype(np.float32)

        # Collect initial inventories from the graph
        initial_inventories = []
        for node in self.graph.nodes:
            if node not in ["S", "D"]:
                initial_inventories.append(self.graph.nodes[node].get("I", 0))

        # Convert to numpy array
        initial_inventories = np.array(initial_inventories, dtype=np.float32).flatten()

        self.state = {
            "inventory_levels": initial_inventories,
            "planned_demand": self.planned_demands,
            "actual_demand": self.current_demand,
        }

        max_lead_time = max([data["L"] for _, _, data in self.graph.edges(data=True)])
        obs = {
            "inventory_levels": self.inventory.astype(np.float32),
            "current_demand": self.actual_demands[0].astype(np.float32),
            "backlog_levels": np.array(
                [len(queue) for queue in self.backlog_queues.values()], dtype=np.float32
            ),
            "order_queues": np.array(
                [
                    list(self.order_queues[node])
                    + [0] * (max_lead_time - len(self.order_queues[node]))
                    for node in self.graph.nodes
                    if node not in ["S", "D"]
                ],
                dtype=np.float32,
            ),
            "lead_times": np.array(
                [
                    len(self.order_queues[node])
                    for node in self.graph.nodes
                    if node not in ["S", "D"]
                ],
                dtype=np.int32,
            ),
        }

        # Resetting history data
        self.inventory = initial_inventories
        self.stock_history = [self.inventory.tolist()]
        self.reward_history = [np.sum(initial_inventories * self.stock_cost * -1)]

        # Placeholder for info
        info = {}

        return obs, info


class MultiDiscreteToBoxWrapper(gym.ActionWrapper):
    """
    Converts a MultiDiscrete action space to a Box action space.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(
            env.action_space, MultiDiscrete
        ), "Environment must have a MultiDiscrete action space."
        self.original_action_space = env.action_space
        self.action_space = Box(
            low=0.0,
            high=1.0,
            shape=self.original_action_space.nvec.shape,
            dtype=np.float32,
        )

    def action(self, action):
        # Convert continuous action (Box) back to discrete (MultiDiscrete)
        discrete_action = np.round(
            action * (self.original_action_space.nvec - 1)
        ).astype(int)
        return discrete_action

    def reverse_action(self, action):
        # Optionally: Map discrete actions back to normalized (Box)
        normalized_action = action / (self.original_action_space.nvec - 1)
        return normalized_action
