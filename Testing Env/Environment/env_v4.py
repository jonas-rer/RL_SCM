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

from collections import deque

# Import stable baselines
# Probably not needed --> test ray lib instead
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class SS_Mngmt_Env(Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Define the action and observation space
    def __init__(self, EP_LENGTH = 30, network_config = None, render_mode = None):

        # To implement
        # handling of safety stock

        self.EP_LENGTH = EP_LENGTH # Total length
        self.episode_length = EP_LENGTH # Current length of the episode

        # Initialize the reward
        self.reward = 0

        # Seting up the network
        self.network_config = network_config
        self.graph = nx.DiGraph()
        self.setup_network(self.network_config)

        # Define the costs
        self.stockout_cost = 10
        self.order_cost = 5
        self.item_cost = 1
        self.stock_cost = 1

        # Order delay and queue
        self.order_queues = self.order_queue()

        # Backlog queue for each node
        self.backlog_queues = self.backlog_queue()
        
        # Define action space
        # The action space is a box with dimensions equal to the number of nodes
        # This represents the order amount for each node
        low = np.full(len(self.graph.nodes) - 2, -1)
        high = np.full(len(self.graph.nodes) - 2, 1)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # Define observation space
        # Number of nodes including 'S' and 'D'
        num_nodes = len(self.graph.nodes) - 2

        # Define the lower bounds for stock level and expected demand
        low_stock = np.full((1, num_nodes), -1)
        low_demand = np.full((self.EP_LENGTH, num_nodes), -1)
        low = np.concatenate([low_stock, low_demand]).flatten()

        # Define the upper bounds for stock level and expected demand
        high_stock = np.full((1, num_nodes), 1)
        high_demand = np.full((self.EP_LENGTH, num_nodes), 1)
        high = np.concatenate([high_stock, high_demand]).flatten()

        self.observation_space = Box(low=low, high=high, dtype=np.float64)

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

        # Dynamic normalization: calculate the observed min and max inventory values
        self.inventory_min = np.min(initial_inventories)
        self.inventory_max = np.max(initial_inventories)

        # Check if all values are the same (to avoid division by zero)
        if self.inventory_max == self.inventory_min:
            normalized_inventories = np.zeros_like(initial_inventories)
        else:
            # Normalize inventories to the range [-1, 1]
            normalized_inventories = 2 * ((initial_inventories - self.inventory_min) / (self.inventory_max - self.inventory_min)) - 1

        # Assuming planned demands have a fixed range [0, 100]
        demand_min = 0
        demand_max = 100
        normalized_planned_demands = 2 * ((self.planned_demands - demand_min) / (demand_max - demand_min)) - 1

        normalized_inventories = normalized_inventories.reshape(1, -1)

        self.state = np.concatenate([normalized_inventories, normalized_planned_demands]).flatten()

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

        # Retrieving inventory levels
        inventory_levels = self.state[:num_nodes]

        # Check if all values are the same (to avoid division by zero)
        if self.inventory_max == self.inventory_min:
            inventory_levels = np.zeros_like(inventory_levels)
        else:
            inventory_levels = (inventory_levels + 1) / 2 * (self.inventory_max - self.inventory_min) + self.inventory_min

        # Retrieve the actual demand for the current timestep
        self.current_demand = self.actual_demands[timestep]

        # Add every first element of the order queues to the history
        self.new_order = ((action + 1) / 2) * 100
        self.reward -= np.sum(self.order_cost * (self.new_order * self.item_cost))

        self.orders = [self.order_queues[node][0] for node in self.graph.nodes if node not in ['S', 'D']]

        for node in self.graph.nodes:
            
            if node not in ['S', 'D']:

                # Get the index of the node
                node_index = self.node_to_index(node)

                # Orders are delivered from the order queues
                # Get the order from the order queue
                order = self.order_queues[node].popleft()

                # Add the order to the stock level
                inventory_levels[node_index] += order

                # Process the backlog first
                while self.backlog_queues[node] and inventory_levels[node_index] > 0:
                    backlog_demand = self.backlog_queues[node][0]
                    if inventory_levels[node_index] >= backlog_demand:
                        inventory_levels[node_index] -= backlog_demand
                        self.backlog_queues[node].popleft()  # Remove the processed demand from the backlog
                    else:
                        break  # Not enough stock to fulfill the backlog, so break the loop

                # If there's still stock left after processing the backlog, process the current demand
                if inventory_levels[node_index] >= self.current_demand[node_index]:
                    inventory_levels[node_index] -= self.current_demand[node_index]
                else:
                    # Add the demand to the backlog queue
                    self.backlog_queues[node].append(self.current_demand[node_index])

                    # Penalty for stockout
                    self.reward -= self.stockout_cost

                # New orders can be placed and will be added to the deque (order_queues)
                # Add the order to the order queue
                self.order_queues[node].append(self.new_order[node_index])

        # Compute the reward based on the order costs and stock level
        self.reward -= np.sum(inventory_levels * self.stock_cost)

        # Check if the episode is done
        done = self.episode_length == 0

        # Decrease the episode length
        self.episode_length -= 1

        self.inventory = inventory_levels

        # Normalize the inventory levels
        self.inventory_min = np.min(inventory_levels)
        self.inventory_max = np.max(inventory_levels)

        normalized_inventories = 2 * ((inventory_levels - self.inventory_min) / (self.inventory_max - self.inventory_min)) - 1

        normalized_inventories = normalized_inventories.reshape(1, -1)

        demand_min = 0
        demand_max = 100
        normalized_planned_demands = 2 * ((self.planned_demands - demand_min) / (demand_max - demand_min)) - 1

        # Update the state
        self.state = np.concatenate([normalized_inventories, normalized_planned_demands]).flatten()

        # Update the observation space
        obs = np.copy(self.state)

        # TODO Does it improve if state[1] the demand is updated with the actual demand for each step?

        # Check if episode is done
        if self.episode_length <= 0: 
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Check if the episode is truncated
        truncated = False

        return obs, float(self.reward), done, truncated, info

    def render(self):
        # Just check episode lenghth and only plot the last one when using matplotlib          
        if self.render_mode is not None:
            if self.render_mode == "human":
                self.render_human()
                # self.render_pygame()

    def render_human(self):

        print(f"Episode Length: {self.EP_LENGTH - self.episode_length}")
        print(f"Stock Level: {self.inventory}")
        print(f"Planned Demand: {self.planned_demands[self.EP_LENGTH - self.episode_length - 1]}")
        print(f"Actual Demand: {self.current_demand}")
        print(f"Action: {self.new_order}")
        print(f"Order: {self.orders}")
        print(f"Reward: {self.reward}")
        print()
              
        self.stock_history.append(list(self.inventory))
        self.demand_history.append(self.current_demand)
        self.action_history.append(self.new_order)
        self.delivery_history.append(self.orders)
        self.reward_history.append(self.reward)

        # Save the data
        now = datetime.now()
        path = f'./Data/{now.strftime("%Y-%m-%d_%H")}_last_environment_data.csv'
        
        self.save_data(path)

        return
    
    def save_data(self, path):

        # Initialize a list to store the data
        data = []

        # Loop over all time steps
        for t in range(len(self.stock_history)):
            # Loop over all nodes
            for n in range(len(self.stock_history[t])):
                # Create a dictionary with the data for this node at this time step
                row = {
                    'Time': t + 1,
                    'Node': self.get_node_name(n),
                    'Stock': self.stock_history[t][n],
                    'Action': self.action_history[t][n],
                    'Demand': self.demand_history[t][n],
                    'Delivery': self.delivery_history[t][n],
                    'Reward': self.reward_history[t]
                }
                # Append the dictionary to the list
                data.append(row)

        # Convert the list of dictionaries to a pandas DataFrame
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
        # Print node attributes
        print("Node Attributes:")
        for node, attributes in self.graph.nodes(data=True):
            print(f"Node {node}: {attributes}")

        # Define the hierarchical layout using graphviz's 'dot'
        pos = graphviz_layout(self.graph, prog="dot")

        # Define the plot
        plt.figure(figsize=(8, 6))

        # Draw the nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='lightblue')

        # Draw the edges
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges(), arrowstyle='->', arrowsize=20)

        # Draw the node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_family="sans-serif")

        # Extract the edge labels (lead times) and draw them
        edge_labels = nx.get_edge_attributes(self.graph, 'L')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        # Set plot title
        plt.title("Supply Chain Network Graph", fontsize=15)

        # Display the plot
        plt.axis('off')  # Turn off the axis
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

        # Create the planned_demand array based on these edges
        planned_demand = np.zeros((self.EP_LENGTH, len(edges_leading_to_D)))

        for i, edge in enumerate(edges_leading_to_D):
            for j in range(self.EP_LENGTH):
                # Introduce a probability of having demand
                if np.random.rand() < 0.5:  # 50% chance of having demand
                    planned_demand[j, i] = np.random.normal(10, 2)

        # TODO make sure that the demand has the right shape

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
                    noise = np.random.normal(0, 5)
                    # Ensure actual demand is not less than 0
                    actual_demand[i, j] = max(0, actual_demand[i, j] + noise)

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

        self.reward = 0

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

        # Dynamic normalization: calculate the observed min and max inventory values
        self.inventory_min = np.min(initial_inventories)
        self.inventory_max = np.max(initial_inventories)

        # Check if all values are the same (to avoid division by zero)
        if self.inventory_max == self.inventory_min:
            normalized_inventories = np.zeros_like(initial_inventories)
        else:
            # Normalize inventories to the range [-1, 1]
            normalized_inventories = 2 * ((initial_inventories - self.inventory_min) / (self.inventory_max - self.inventory_min)) - 1

        # Assuming planned demands have a fixed range [0, 100]
        demand_min = 0
        demand_max = 100
        normalized_planned_demands = 2 * ((self.planned_demands - demand_min) / (demand_max - demand_min)) - 1

        normalized_inventories = normalized_inventories.reshape(1, -1)

        # Update the state
        self.state = np.concatenate([normalized_inventories, normalized_planned_demands]).flatten()

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