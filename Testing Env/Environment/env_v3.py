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
    def __init__(self, EP_LENGTH=30, network_config=None, render_mode = None):

        # To implement
        # storage capacities
        # safety stock

        # Backlog
        # Production time! --> meaning each stock has different stocks... raw material and product
        # Could be implememted with an additional deque for each node. Where new material gets added to the and the oldest gets removed and added to the inventory

        self.EP_LENGTH = EP_LENGTH # Total length
        self.episode_length = EP_LENGTH # Current length of the episode

        # Seting up the network
        self.network_config = network_config
        self.graph = nx.DiGraph()
        self.setup_network()

        # Order delay and queue
        # Initialize an empty dictionary for the order queues
        self.order_queues = {}

        # # For each node in the network, create a queue and add it to the dictionary
        # for node in self.graph.nodes:
        #     self.order_queues[node] = deque(maxlen=self.lead_time)

        # For each node in the network, create a queue and add it to the dictionary
        for node in self.graph.nodes:
            # Get the lead time for the node from the edge leading to it
            in_edges = list(self.graph.in_edges(node, data=True))
            if in_edges:  # Check if the list is not empty
                lead_time = in_edges[0][2]['L']
                self.order_queues[node] = deque(maxlen=lead_time)

        # Demand
        # TODO change that demand comes from a distribution
        # TODO allow for loading the planned demand from a file
        
        # Setting up the demand
        self.demand = 0
        self.expected_demand = np.zeros(len(self.graph.nodes))

        # Define action space
        # The action space is a box with dimensions equal to the number of nodes
        # This represents the order amount for each node
        low = np.zeros(len(self.graph.nodes))  # minimum order amount for each node
        high = np.full(len(self.graph.nodes), 100)  # maximum order amount for each node

        self.action_space = Box(low=low, high=high, dtype=np.int64)

        # Define observation space
        # This represents the stock level and expected demand for each node
        low = np.concatenate([np.zeros(len(self.graph.nodes)), np.zeros(len(self.graph.nodes))])
        max_stock = np.full(len(self.graph.nodes), 100)
        max_demand = np.full(len(self.graph.nodes), 100)
        high = np.concatenate([max_stock, max_demand])

        self.observation_space = Box(low=low, high=high, dtype=np.int64)

        # Define the initial state
        self.state = {}

        initial_inventories = []
        expected_demands = []

        for node in self.graph.nodes:
            initial_inventories.append(self.graph.nodes[node].get('I', 0))
            # Get the edges that have the current node as their target
            in_edges = list(self.graph.in_edges(node, data=True))
            # If there are any such edges, get the 'D' value from the first one
            if in_edges:
                expected_demands.append(in_edges[0][2].get('D', 0))
            else:
                expected_demands.append(0)

        # Convert the lists to a numpy array
        self.state = np.array([initial_inventories] + [expected_demands])

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

        # TODO: Demand can only be fullfilled if sufficient stock is available demand is then placed in the backlog queue plus a penalty for stockout

        # Expected demand for production at timestep t + lead time
        i = self.EP_LENGTH - self.episode_length
        if i + self.lead_time < len(self.planned_demand):
            self.state[1] = self.planned_demand[i + self.lead_time]

        # Demand for the current timestep
        if self.planned_demand[i] > 0:
            # Calculate the actual demand with a random factor
            self.demand = self.planned_demand[i] + random.randint(-3, 5)
        else:
            self.demand = 0

        # Subtract the demand from the stock
        self.state[0] = self.state[0] - self.demand

        # For each edge in the network
        for edge in self.graph.edges:

            # Expected demand for production at timestep t + lead time
            i = self.EP_LENGTH - self.episode_length
            if i + self.graph.edges[edge]['L'] < len(self.graph.edges[edge]['D']):
                self.state[edge][1] = self.graph.edges[edge]['D'][i + self.graph.edges[edge]['L']]

            # Demand for the current timestep
            if self.graph.edges[edge]['D'][i] > 0:
                # Calculate the actual demand with a random factor 
                self.demand[edge] = self.graph.edges[edge]['D'][i] + random.randint(-3, 5)
            else:
                self.demand[edge] = 0

            # Subtract the demand from the stock
            self.state[edge][0] = self.state[edge][0] - self.demand[edge]

        # Iterate over each node in the network
        for node in self.graph.nodes:
            # Get the lead time for the node
            lead_time = self.graph.in_edges(node, data=True)[0][2]['L']

            # If there are enough steps passed since the order was placed, add the order to the stock
            if len(self.order_queues[node]) == lead_time:
                self.delivered[node] = self.order_queues[node].popleft()
                self.state[node][0] += self.delivered[node]
                
            # Add the order to the queue
            # TODO: This needs to be changed will not work
            self.order_queues[node].append(action[node])

        # Calculate the reward based on order costs
        if action > 0:
            reward = - self.c_fixed - self.c_variable * action
        else:
            reward = 0

        # Check if the stock level is negative
        if self.state[0] < 0:
            # If the stock level is negative, the added cost is the stockout cost
            # TODO: The order can only be fullfilled on the next timestep
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
        # LEGACY CODE

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
    
    def setup_network(self):
        # Load the network configuration from a JSON string
        config = json.loads(self.network_config)

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

    def planned_demand(self):

        # Genrates a random planned demand for each edge in the network
        # over the whole episode. The demand is drawn from a normal distribution

        planned_demand = np.zeros((self.EP_LENGTH, len(self.graph.edges)))

        for edge in self.graph.edges:
            planned_demand[:, edge] = np.random.normal(10, 2, self.EP_LENGTH)

        return planned_demand
    
    def actual_demand(self):

        # Genrate a random actual demand for each edge in the network
        # over the whole episode. The demand is drawn from a normal distribution

        actual_demand = np.zeros((self.EP_LENGTH, len(self.graph.edges)))

        for edge in self.graph.edges:
            actual_demand[:, edge] = np.random.normal(10, 2, self.EP_LENGTH)

        return actual_demand


    def reset(self, seed = None):
        # Reset the state of the environment back to an initial state

        super().reset(seed = seed) # Reset the seed
        if seed is not None:
            random.seed(seed)

        # Reset the episode length
        self.episode_length = self.EP_LENGTH

        # Reset the network
        self.setup_network()

        # Demand
        self.expected_demand = 0 # Expected demand for production
        self.delivered = 0

        # Reset the order queue
        self.order_queue.clear()

        initial_inventories = []
        expected_demands = []

        for node in self.graph.nodes:
            initial_inventories.append(self.graph.nodes[node].get('I', 0))
            # Get the edges that have the current node as their target
            in_edges = list(self.graph.in_edges(node, data=True))
            # If there are any such edges, get the 'D' value from the first one
            if in_edges:
                expected_demands.append(in_edges[0][2].get('D', 0))
            else:
                expected_demands.append(0)

        # Setting state and observation
        self.state = np.array([initial_inventories] + [expected_demands])
        obs = np.array([initial_inventories] + [expected_demands])

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
