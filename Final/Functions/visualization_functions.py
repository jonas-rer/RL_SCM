# Import helpers
import numpy as np
import pandas as pd
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


# Function to plot the data
def plot_data(df):

    # Get the unique nodes
    nodes = df["Node"].unique()

    # Create a list of colors
    colors = ["b", "g", "c", "m", "y", "k"]

    # Create a figure with a subplot for each node and each variable
    fig, axs = plt.subplots(len(nodes), 4, figsize=(20, len(nodes) * 5))

    # Loop over all nodes
    for i, node in enumerate(nodes):
        # Select the data for this node
        node_data = df[df["Node"] == node]

        # Plot the 'Stock' over time
        axs[i, 0].bar(
            node_data["Time"],
            node_data["Stock"],
            label=f"Node: {node}",
            color=colors[i % len(colors)],
        )

        axs[i, 0].set_title(f"Stock over time for node {node}")

        mean_stock = node_data["Stock"].mean()
        axs[i, 0].axhline(
            mean_stock,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_stock:.2f}",
        )

        # Plot the 'Delivery' over time
        axs[i, 1].bar(
            node_data["Time"],
            node_data["Delivery"],
            label=f"Node: {node}",
            color=colors[i % len(colors)],
        )

        mean_delivery = node_data["Delivery"].mean()
        axs[i, 1].axhline(
            mean_delivery,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_delivery:.2f}",
        )

        axs[i, 1].set_title(f"Delivery over time for node {node}")

        # Plot the 'Demand' over time
        axs[i, 2].bar(
            node_data["Time"],
            node_data["Demand"],
            label=f"Node: {node}",
            color=colors[i % len(colors)],
        )
        axs[i, 2].set_title(f"Demand over time for node {node}")

        mean_demand = node_data["Demand"].mean()
        axs[i, 2].axhline(
            mean_demand,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_demand:.2f}",
        )

    # Plots the reward over time and the total reward
    # Plot the 'Reward' over time
    axs[0, 3].plot(df["Time"], df["Reward"], label="Reward per Timestep", color="r")
    axs[0, 3].set_title("Reward over time")

    # Plot the total reward over time
    axs[1, 3].plot(df["Time"], df["Total Reward"], label="Total Reward", color="r")
    axs[1, 3].set_title("Total Reward over time")

    # Plot stockouts over time
    stockouts = df["Backlog"].sum()
    axs[2, 3].plot(df["Time"], df["Backlog"], label="Stockouts", color="r")
    axs[2, 3].set_title(f"Stockouts over time: {stockouts}")

    # Add a legend to the plot
    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[2, 0].legend()

    # Add more space between the plots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()


# Function to visualize the stock and safety stock


def plot_safety_stock(df):

    # Compute the safety stock level for each node
    safety_stock = df["Demand"].mean() * df["Lead Time"].mean()

    # Get the unique nodes
    nodes = df["Node"].unique()

    # Create a list of colors
    colors = ["b", "g", "c", "m", "y", "k"]

    # Create a figure with a subplot for each node and each variable
    fig, axs = plt.subplots(len(nodes), 4, figsize=(20, len(nodes) * 5))

    # Loop over all nodes
    for i, node in enumerate(nodes):
        # Select the data for this node
        node_data = df[df["Node"] == node]

        # Plot the 'Stock' over time
        axs[i, 0].bar(
            node_data["Time"],
            node_data["Stock"],
            label=f"Node: {node}",
            color=colors[i % len(colors)],
        )

        axs[i, 0].set_title(f"Stock over time for node {node}")

        mean_stock = node_data["Stock"].mean()
        axs[i, 0].axhline(
            mean_stock,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_stock:.2f}",
        )
