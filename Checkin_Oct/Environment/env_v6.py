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
from datetime import datetime
from pprint import pprint

from collections import deque

# Import stable baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class SS_Mngmt_Env(Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Define the action and observation space
    def __init__(
                self, 
                EP_LENGTH = 52, 
                network_config = None,
                render_mode = None,
                stockout_cost = 1000,
                order_cost = 5,
                item_cost = 0.1,
                stock_cost = 0.5,
                item_prize = 20,
                order_quantities = [0, 15, 50]):

        self.EP_LENGTH = EP_LENGTH # Total length
        self.episode_length = EP_LENGTH # Current length of the episode

        # Seting up the network
        self.network_config = network_config
        self.graph = nx.DiGraph()
        self.setup_network(self.network_config)

        # Number of nodes excluding 'S' and 'D'
        num_nodes = len(self.graph.nodes) - 2

        # Define the costs
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.item_cost = item_cost
        self.stock_cost = stock_cost
        self.item_prize = item_prize

        self.order_quantities = order_quantities

        # Order delay and queue
        self.order_queues = self.order_queue()

        # Backlog queue for each node
        self.backlog_queues = self.backlog_queue()
        
        # Define action space
        n_actions = 3
        n_nodes = len(self.graph.nodes) - 2
        action_choices = np.full(n_nodes, n_actions)
        self.action_space = MultiDiscrete(action_choices)

        # Define the lower bounds for stock level and expected demand
        low_stock = np.full((1, num_nodes), 0)
        low_demand = np.full((self.EP_LENGTH, num_nodes), 0)
        low = np.concatenate([low_stock, low_demand]).flatten()

        # Define the upper bounds for stock level and expected demand
        high_stock = np.full((1, num_nodes), 1000)
        high_demand = np.full((self.EP_LENGTH, num_nodes), 20)
        high = np.concatenate([high_stock, high_demand]).flatten()

        # Define the observation space
        self.observation_space = Box(low=low, high=high, dtype=np.float64)

        # Define the initial state
        self.planned_demands = self.planned_demand()
        self.actual_demands = self.actual_demand(self.planned_demands)

        # Collect initial inventories from the graph
        initial_inventories = []
        for node in self.graph.nodes:
            if node not in ['S', 'D']:
                initial_inventories.append(self.graph.nodes[node].get('I', 0))

        initial_inventories = np.array(initial_inventories)
        initial_inventories = initial_inventories.reshape(1, initial_inventories.shape[0])

        self.state = np.concatenate([initial_inventories, self.planned_demands]).flatten()

        # Prep to save the data
        self.inventory = initial_inventories
        self.stock_history = self.inventory.tolist()
        self.action_history = [np.zeros(num_nodes)]
        self.demand_history = [np.zeros(num_nodes)]
        self.delivery_history = [np.zeros(num_nodes)]
        self.reward_history = [0]

        # Render mode
        self.render_mode = render_mode
        self.screen_initialized = False


    # Defining the step function
    def step(self, action):

        # Returns the next state, reward and whether the episode is done
        timestep = self.EP_LENGTH - self.episode_length

        num_nodes = len(self.graph.nodes) - 2

        # Retrieve the current inventory levels
        self.inventory = self.state[:num_nodes]
        inventory_levels = np.copy(self.inventory)

        reward = 0

        # Retrieve the actual demand for the current timestep
        self.current_demand = self.actual_demands[timestep]

        # Add every first element of the order queues to the history
        self.new_order = []

        for i in action:
            self.new_order.append(self.order_quantities[i])

        # For visualization and history data
        self.orders = np.array([self.order_queues[node][0] for node in self.graph.nodes if node not in ['S', 'D']])

        # Process the orders and update the inventory levels for each node
        for node in self.graph.nodes:
            
            if node not in ['S', 'D']:

                # Get the index of the node
                node_index = self.node_to_index(node)

                if self.new_order[node_index] > 0:
                    reward -= self.order_cost + (self.new_order[node_index] * self.item_cost)

                # Get the order from the order queue, add it to stock level
                order = self.order_queues[node].popleft()
                inventory_levels[node_index] += order

                # If there's still stock left after processing the backlog, process the current demand
                node_demand = self.current_demand[node_index]
                if inventory_levels[node_index] >= node_demand:
                    inventory_levels[node_index] -= node_demand
                    reward += node_demand * self.item_prize  # Reward for fulfilling the demand
                else:
                    # Add the demand to the backlog queue
                    self.backlog_queues[node].append(node_demand)

                    # Penalty for stockout
                    reward -= self.stockout_cost

                # Process the backlog
                while self.backlog_queues[node] and inventory_levels[node_index] > 0:
                    backlog_demand = self.backlog_queues[node][0]
                    if inventory_levels[node_index] >= backlog_demand:
                        inventory_levels[node_index] -= backlog_demand
                        reward += backlog_demand * self.item_prize # Reward for fulfilling the backlog
                        self.backlog_queues[node].popleft()  # Remove the processed demand from the backlog
                    else:
                        break  # Not enough stock to fulfill the backlog, so break the loop

                # Add the order to the order queue
                self.order_queues[node].append(self.new_order[node_index])

        # Compute the reward based on the order costs and stock level
        reward -= np.sum(inventory_levels * self.stock_cost)

        # Check if the episode is done
        done = self.episode_length == 0

        # Decrease the episode length
        self.episode_length -= 1

        inventory_levels = inventory_levels.reshape(1, inventory_levels.shape[0])
        self.inventory = inventory_levels

        self.state = np.concatenate([self.inventory, self.actual_demands]).flatten()

        # Update the observation space
        obs = np.copy(self.state)

        self.reward_history.append(reward)

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

    def render_human(self):

        try:

            print(f"Episode Length: {self.EP_LENGTH - self.episode_length}")
            print(f"Stock Level: {self.inventory}")
            print(f"Planned Demand: {self.planned_demands[self.EP_LENGTH - self.episode_length - 1]}")
            print(f"Actual Demand: {self.current_demand}")
            print(f"Action: {self.new_order}")
            print(f"Order: {self.orders}")
            print(f"Reward: {self.reward_history[self.EP_LENGTH - self.episode_length - 1]}")
            print()
            print("Backlog:")
            pprint(self.backlog_queues, indent=4)

            print("Order Queue:")
            pprint(self.order_queues, indent=4)
            print()
                
            self.stock_history.append(self.inventory[0])
            self.demand_history.append(self.current_demand)
            self.action_history.append(self.new_order)
            self.delivery_history.append(self.orders)

            # Save the data
            now = datetime.now()
            path = f'./Data/{now.strftime("%Y-%m-%d_%H")}_last_environment_data.csv'
            
            self.save_data(path)

        except Exception as e:
            print()

        return
    
    def save_data(self, path):
        # Saves the episode data to a CSV file

        data = []

        for t in range(len(self.stock_history)):
            for n in range(len(self.stock_history[t])):
                row = {
                    'Time': t + 1,
                    'Node': self.get_node_name(n),
                    'Stock': self.stock_history[t][n],
                    'Action': self.action_history[t][n],
                    'Demand': self.demand_history[t][n],
                    'Delivery': self.delivery_history[t][n],
                    'Reward': self.reward_history[t]
                }
                data.append(row)

        df = pd.DataFrame(data)

        if self.episode_length == 1:
            df.to_csv(path, index=False)

    def setup_network(self, network_config = None):
        # Load the network configuration from a JSON string
        config = json.loads(network_config)

        # Add nodes to the graph
        for node, attributes in config['nodes'].items():
            self.graph.add_node(node, **attributes)

        # Add edges to the graph with lead times
        for edge in config['edges']:
            self.graph.add_edge(edge['source'], edge['target'], L=edge['L'])

    def render_network(self):
        # Render the network using networkx

        print("Node Attributes:")
        for node, attributes in self.graph.nodes(data=True):
            print(f"Node {node}: {attributes}")

        pos = graphviz_layout(self.graph, prog="dot")

        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges(), arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_family="sans-serif")

        edge_labels = nx.get_edge_attributes(self.graph, 'L')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Supply Chain Network Graph", fontsize=15)

        # Display the plot
        plt.axis('off') 
        plt.show()

    def node_to_index(self, node):
        # Creates a mapping from node names to indices
        return list(self.graph.nodes).index(node)
    
    def get_node_name(self, index):
        # Creates a mapping from indices to node names
        return list(self.graph.nodes)[index]

    def planned_demand(self):
        # Generates a random planned demand for each edge in the network
        # over the whole episode. The demand is drawn from a normal distribution
        edges_leading_to_D = [edge for edge in self.graph.edges if edge[1] == 'D']

        planned_demand = np.zeros((self.EP_LENGTH, len(edges_leading_to_D)))

        for i, edge in enumerate(edges_leading_to_D):
            for j in range(self.EP_LENGTH):
                # Introduce a probability of having demand
                if np.random.rand() < 0.5:  # 50% chance of having demand
                    planned_demand[j, i] = int(np.random.normal(10, 3))

        return planned_demand
    
    def actual_demand(self, planned_demand):

        # Generate a random actual demand for each edge in the network
        # based on the planned demand from the current timestep. The demand
        # is drawn from a normal distribution
        actual_demand = np.copy(planned_demand)

        for i in range(actual_demand.shape[0]):
            for j in range(actual_demand.shape[1]):
                # Add a small random noise to the planned demand
                if planned_demand[i, j] > 0:
                    noise = np.random.normal(0, 2)
                    # Ensure actual demand is not less than 0
                    actual_demand[i, j] = int(max(0, actual_demand[i, j] + noise))

        return actual_demand
    
    def order_queue(self):
        # Create a dictionary for the order queues
        order_queues = {}

        for node in self.graph.nodes:

            if node not in ['S', 'D']:
                in_edges = list(self.graph.in_edges(node, data=True))

                if in_edges:
                    lead_time = in_edges[0][2]['L']
                    order_queues[node] = deque(maxlen=lead_time)

                    order_queues[node].extend([0] * lead_time)

        return order_queues
    
    def backlog_queue(self):
        # Create a dictionary for the backlog queues
        backlog_queues = {}

        for node in self.graph.nodes:
            if node not in ['S', 'D']:

                in_edges = list(self.graph.in_edges(node, data=True))
                if in_edges:
                    backlog_queues[node] = deque()

        return backlog_queues
    
    def reset(self, seed = None):
        # Reset the state of the environment back to an initial state

        super().reset(seed = seed) # Reset the seed
        if seed is not None:
            random.seed(seed)

        # Reset the episode length
        self.episode_length = self.EP_LENGTH

        # Reset the network
        self.graph = nx.DiGraph()
        self.setup_network(self.network_config)

        num_nodes = len(self.graph.nodes) - 2

        # Order delay and backlog queue
        self.order_queues = self.order_queue()
        self.backlog_queues = self.backlog_queue()

        # Define the initial state
        self.planned_demands = self.planned_demand()
        self.actual_demands = self.actual_demand(self.planned_demands)

        # Collect initial inventories from the graph
        initial_inventories = []
        for node in self.graph.nodes:
            if node not in ['S', 'D']:
                initial_inventories.append(self.graph.nodes[node].get('I', 0))

        # Convert to numpy array
        initial_inventories = np.array(initial_inventories)
        initial_inventories = initial_inventories.reshape(1, initial_inventories.shape[0])

        # Update the state
        self.state = np.concatenate([initial_inventories, self.planned_demands]).flatten()

        obs = np.copy(self.state)

        # Resetting history data
        self.inventory = initial_inventories
        self.stock_history = self.inventory.tolist()
        self.action_history = [np.zeros(num_nodes)]
        self.demand_history = [np.zeros(num_nodes)]
        self.delivery_history = [np.zeros(num_nodes)]
        self.reward_history = [0]

        # Placeholder for info
        info = {}

        return obs, info