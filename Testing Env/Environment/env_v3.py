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
        self.stock_cost = 0.5

        # Order delay and queue
        # Initialize an empty dictionary for the order queues
        self.order_queues = {}

        # For each node in the network, create a queue and add it to the dictionary
        for node in self.graph.nodes:
            # Get the lead time for the node from the edge leading to it
            in_edges = list(self.graph.in_edges(node, data=True))
            if in_edges:  # Check if the list is not empty
                lead_time = in_edges[0][2]['L']
                self.order_queues[node] = deque(maxlen=lead_time)

                # Fill the queue with zeros
                self.order_queues[node].extend([0] * lead_time)

        # Backlog queue for each node
        self.backlog_queues = {}

        # For each node in the network, create a queue and add it to the dictionary
        for node in self.graph.nodes:
            # Get the lead time for the node from the edge leading to it
            in_edges = list(self.graph.in_edges(node, data=True))
            if in_edges:
                self.backlog_queues[node] = deque()
        
        # Define action space
        # The action space is a box with dimensions equal to the number of nodes
        # This represents the order amount for each node
        low = np.zeros(len(self.graph.nodes) - 2)  # minimum order amount for each node
        high = np.full(len(self.graph.nodes) - 2, 100)  # maximum order amount for each node

        self.action_space = Box(low=low, high=high, dtype=np.int64)

        # Define observation space
        # Number of nodes including 'S' and 'D'
        num_nodes = len(self.graph.nodes) - 2

        # Define the lower bounds for stock level and expected demand
        low_stock = np.zeros((1, num_nodes))
        low_demand = np.zeros((self.EP_LENGTH - 1, num_nodes))
        low = np.concatenate([low_stock, low_demand])

        # Define the upper bounds for stock level and expected demand
        high_stock = np.full((1, num_nodes), 100)
        high_demand = np.full((self.EP_LENGTH - 1, num_nodes), 100)
        high = np.concatenate([high_stock, high_demand])

        self.observation_space = Box(low=low, high=high, dtype=np.float64)

        # Define the initial state
        self.planned_demands = self.planned_demand()

        self.actual_demands = self.actual_demand(self.planned_demands)

        initial_inventories = []
        for node in self.graph.nodes:

            if node not in ['S', 'D']:
                initial_inventories.append(self.graph.nodes[node].get('I', 0))

        initial_inventories = np.array(initial_inventories)
        initial_inventories = initial_inventories.reshape(1, initial_inventories.shape[0])   

        # Now you can combine them
        self.state = np.concatenate([initial_inventories, self.planned_demands])

        # TODO
        # Empty dataframe for plotting the history and analysis
        # self.history = pd.DataFrame(columns = ['Stock Level', 'Order', 'Demand'])
        # Probably makes sense to have as a separate method

        # History
        self.reward_history = []
        self.stock_history = []
        self.demand_history = []
        self.expected_demand_history = []
        self.order_history = []
        self.delivery_history = []

        # Render mode
        self.render_mode = render_mode
        self.screen_initialized = False


    # Defining the step function
    def step(self, action):
        # Returns the next state, reward and whether the episode is done

        # Retrieve the actual demand for the current timestep
        self.current_demand = self.actual_demands[self.EP_LENGTH - self.episode_length]

        # Subtract the demand from the stock level of the corresponding nodes of the state
        # Leave the demand in the deque for the next timestep if the stock level is negative at first position

        for node in self.graph.nodes:
            
            if node not in ['S', 'D']:

                # Get the index of the node
                node_index = self.node_to_index(node)

                # Orders are delivered from the order queues
                # Get the order from the order queue
                order = self.order_queues[node].popleft()

                # Add the order to the stock level
                self.state[0][node_index] += order

                # Add the order to the delivered list
                self.delivery_history.append(order)

                # Process the backlog first
                while self.backlog_queues[node] and self.state[0][node_index] > 0:
                    backlog_demand = self.backlog_queues[node][0]
                    if self.state[0][node_index] >= backlog_demand:
                        self.state[0][node_index] -= backlog_demand
                        self.backlog_queues[node].pop(0)  # Remove the processed demand from the backlog
                    else:
                        break  # Not enough stock to fulfill the backlog, so break the loop

                # If there's still stock left after processing the backlog, process the current demand
                if self.state[0][node_index] >= self.current_demand[node_index]:
                    self.state[0][node_index] -= self.current_demand[node_index]
                else:
                    # Add the demand to the backlog queue
                    self.backlog_queues[node].append(self.current_demand[node_index])

                    # Penalty for stockout
                    self.reward -= self.stockout_cost

                # New orders can be placed and will be added to the deque (order_queues)
                # Add the order to the order queue
                self.order_queues[node].append(action[node_index])

                # Compute the order cost
                self.reward -= self.order_cost * (order * self.item_cost)

        # Compute the reward based on the order costs and stock level
        self.reward -= np.sum(self.state[0] * self.stock_cost)

        # Check if the episode is done
        done = self.episode_length == 0

        # Decrease the episode length
        self.episode_length -= 1

        # Update the observation space
        # obs = np.concatenate([self.state[0].reshape(1, -1), self.planned_demands[self.EP_LENGTH - self.episode_length:]])
        obs = np.concatenate([self.state[0].reshape(1, self.state[0].shape[0]), self.planned_demands])

        # TODO Does it improve if state[1] the demand is updated with the actual demand for each step?

        # Append the state to the history
        self.reward_history.append(self.reward)
        self.stock_history.append(self.state[0])
        self.order_history.append(action)
        self.demand_history.append(self.demand)

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

        # To Do
        # LEGACY CODE

        # Change that it prints the states of each node (inventory etc.)

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
        ax[0, 0].set_xticks(np.arange(0, 40, step=5))
        ax[0, 0].set_yticks(np.arange(0, 105, step=10))

        # Plotting the order, delivered and demand
        ax[1, 0].bar(timestamps - bar_width, self.order_history, bar_width, color='blue', alpha = 0.5, label='Order')
        ax[1, 0].bar(timestamps, self.delivery_history, bar_width, color='red', alpha = 0.5, label='Delivery')
        ax[1, 0].bar(timestamps + bar_width, self.demand_history, bar_width, color='green', alpha=0.5, label='Demand')
        ax[1, 0].set_ylabel('Order/Delivery/Demand')
        ax[1, 0].set_title('Order vs. Delivery vs. Demand')
        ax[1, 0].set_xticks(np.arange(0, 40, step=5))
        ax[1, 0].legend(['Order', 'Delivery', 'Demand'])

        # Plotting the reward per timestep
        ax[0, 1].plot(self.reward_history, color = 'purple', label = 'Reward')
        ax[0, 1].set_ylabel('Reward')
        # ax[0, 1].set_xlabel('Time')
        ax[0, 1].set_title('Reward vs. Time')
        ax[0, 1].set_xticks(np.arange(0, 40, step=5))

        # Plotting the accumulated reward
        ax[1, 1].plot(np.cumsum(self.reward_history), color = 'orange', label = 'Accumulated Reward')
        ax[1, 1].set_ylabel('Accumulated Reward')
        # ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_title('Accumulated Reward vs. Time')
        ax[1, 1].set_xticks(np.arange(0, 40, step=5))

        # Plotting the expected demand and actual demand
        # TODO --> Shift the expected demand by 3 timesteps and show only the difference
        # ax[2, 0].plot(self.expected_demand_history, color = 'blue', label = 'Expected Demand')
        # ax[2, 0].plot(self.demand_history, color = 'green', label = 'Actual Demand')
        # ax[2, 0].set_ylabel('Demand')
        # ax[2, 0].set_title('Expected vs. Actual Demand')
        # ax[2, 0].set_xticks(np.arange(0, 40, step=5))

        # Shift the expected demand by 3 timesteps to the right
        shifted_expected_demand = np.roll(self.expected_demand_history, 3)

        # Compute the difference between the expected and actual demand
        demand_difference = shifted_expected_demand - self.demand_history

        # Plotting the shifted expected demand and actual demand
        ax[2, 0].plot(shifted_expected_demand, color = 'blue', label = 'Expected Demand')
        ax[2, 0].plot(self.demand_history, color = 'green', label = 'Actual Demand')
        ax[2, 0].set_ylabel('Demand')
        ax[2, 0].set_title('Expected vs. Actual Demand')
        ax[2, 0].set_xticks(np.arange(0, 40, step=5))

        # Plotting the difference between the expected and actual demand
        ax[2, 1].plot(demand_difference, color = 'red', label = 'Demand Difference')
        ax[2, 1].set_ylabel('Demand Difference')
        ax[2, 1].set_title('Difference between Expected and Actual Demand')
        ax[2, 1].set_xticks(np.arange(0, 40, step=5))

        # Show the plot if the episode is done
        if self.episode_length == 0:
            # plt.tight_layout()
            plt.show()
        
        return
    
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

    def planned_demand(self):
        # Generates a random planned demand for each edge in the network
        # over the whole episode. The demand is drawn from a normal distribution
        edges_leading_to_D = [edge for edge in self.graph.edges if edge[1] == 'D']

        # Create the planned_demand array based on these edges
        planned_demand = np.zeros((self.EP_LENGTH - 1, len(edges_leading_to_D)))

        for i, edge in enumerate(edges_leading_to_D):
            for j in range(self.EP_LENGTH - 1):
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

        # Order delay and queue
        # Initialize an empty dictionary for the order queues
        self.order_queues = {}

        # For each node in the network, create a queue and add it to the dictionary
        for node in self.graph.nodes:
            # Get the lead time for the node from the edge leading to it
            in_edges = list(self.graph.in_edges(node, data=True))
            if in_edges:  # Check if the list is not empty
                lead_time = in_edges[0][2]['L']
                self.order_queues[node] = deque(maxlen=lead_time)

        # TODO reset actual and planned demand

        # Define the initial state
        self.planned_demands = self.planned_demand()

        self.actual_demands = self.actual_demand(self.planned_demands)

        initial_inventories = []
        for node in self.graph.nodes:

            if node not in ['S', 'D']:
                initial_inventories.append(self.graph.nodes[node].get('I', 0))

        initial_inventories = np.array(initial_inventories)
        initial_inventories = initial_inventories.reshape(1, initial_inventories.shape[0])   

        self.state = np.concatenate([initial_inventories, self.planned_demands])

        obs = np.copy(self.state)

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
