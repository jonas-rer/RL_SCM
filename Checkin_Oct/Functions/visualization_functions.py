# Import helpers
import numpy as np
import pandas as pd
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df):

    # Get the unique nodes
    nodes = df['Node'].unique()

    # Create a list of colors
    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    # Create a figure with a subplot for each node and each variable
    fig, axs = plt.subplots(len(nodes), 4, figsize=(20, len(nodes)*5))

    # Loop over all nodes
    for i, node in enumerate(nodes):

        node_data = df[df['Node'] == node]
        
        axs[i, 0].bar(node_data['Time'], node_data['Stock'], label=node, color=colors[i % len(colors)])

        axs[i, 0].set_title(f'Stock over time for node {node}')
        
        axs[i, 1].plot(node_data['Time'], node_data['Reward'], label=node, color=colors[i % len(colors)])
        axs[i, 1].set_title(f'Reward over time for node {node}')
        
        axs[i, 2].bar(node_data['Time'], node_data['Delivery'], label=node, color=colors[i % len(colors)])

        mean_delivery = node_data['Delivery'].mean()
        axs[i, 2].axhline(mean_delivery, color='red', linestyle='--', label=f'Mean Delivery: {mean_delivery:.2f}')

        axs[i, 2].set_title(f'Delivery over time for node {node}')
        
        axs[i, 3].bar(node_data['Time'], node_data['Demand'], label=node, color=colors[i % len(colors)])
        axs[i, 3].set_title(f'Demand over time for node {node}')

        mean_demand = node_data['Demand'].mean()
        axs[i, 3].axhline(mean_demand, color='red', linestyle='--', label=f'Mean Demand: {mean_demand:.2f}')

    plt.subplots_adjust(hspace=0.5)

    plt.show()

def plot_agg_data(df):

    # Get the unique nodes
    nodes = df['Node'].unique()

    agg_df = df.groupby('Time').sum().reset_index()
    agg_df['Reward'] = agg_df['Reward'] / len(nodes)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].bar(agg_df['Time'], agg_df['Stock'], color='b')
    axs[0].set_title('Aggregated Stock over time')

    axs[1].plot(agg_df['Time'], agg_df['Reward'], color='g')
    axs[1].set_title('Aggregated Reward over time')

    axs[2].bar(agg_df['Time'], agg_df['Delivery'], color='r')
    axs[2].set_title('Aggregated Delivery over time')

    axs[3].bar(agg_df['Time'], agg_df['Demand'], color='c')
    axs[3].set_title('Aggregated Demand over time')

    plt.subplots_adjust(wspace=0.5)

    # Show the plot
    plt.show()