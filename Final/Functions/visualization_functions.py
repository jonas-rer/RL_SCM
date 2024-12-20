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
    """
    Plot the data in the dataframe.
    """

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
    for i in range(len(nodes)):
        axs[i, 0].legend()

    # Add more space between the plots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()


def plot_safety_stock(df, safety_stock=None):
    """
    Function to plot the stock and safety stock over time for each node
    """

    # Get the unique nodes
    nodes = df["Node"].unique()

    # Create a list of colors
    colors = ["b", "g", "c", "m", "y", "k"]

    # Create a figure with a subplot for each node
    fig, axs = plt.subplots(len(nodes), 1, figsize=(10, len(nodes) * 5))

    # Compute safety stock for each node
    def compute_safety_stock(df):
        safety_stock = []
        for node in df["Node"].unique():
            node_data = df[df["Node"] == node]
            safety_stock.append(node_data[node_data["Action"] > 0]["Stock"].mean())

        return safety_stock

    if safety_stock is not None:
        safety_stock = list(safety_stock.values())
    else:
        safety_stock = compute_safety_stock(df)

    # Loop over all nodes
    for i, node in enumerate(nodes):
        # Select the data for this node
        node_data = df[df["Node"] == node]

        # Plot the 'Stock' over time
        axs[i].bar(
            node_data["Time"],
            node_data["Stock"],
            label=f"Node: {node}",
            color=colors[i % len(colors)],
        )

        axs[i].set_title(f"Stock over time for node {node}")

        if safety_stock is not None:

            axs[i].axhline(
                safety_stock[i],
                color="orange",
                linestyle="--",
                label=f"Safety Stock: {safety_stock[i]:.1f}",
            )

        else:
            # Add the safety stock to the plot
            safety_stock = compute_safety_stock(df)
            axs[i].axhline(
                safety_stock[i],
                color="orange",
                linestyle="--",
                label=f"Safety Stock: {safety_stock[i]:.1f}",
            )

        # cycle_stock = 0.5 * node_data["Stock"].mean() + safety_stock[i]
        # axs[i].axhline(
        #     cycle_stock,
        #     color="red",
        #     linestyle="--",
        #     label=f"Cycle stock: {cycle_stock:.1f}",
        # )

        # Add a legend to the plot
        axs[i].legend()


# Function to compare results of different runs
def benchmark_plot(dataframes, labels):
    """
    Function to compare the results of n different runs.

    Parameters:
    - dataframes: list of pandas DataFrames, each containing the data for a run.
    - labels: list of strings, labels for the corresponding DataFrames.
    """
    num_runs = len(dataframes)
    colors = plt.cm.tab10(
        np.linspace(0, 1, num_runs)
    )  # Generate distinct colors for each run

    # Create a figure with a subplot for each variable
    fig, axs = plt.subplots(
        1, 3, figsize=(20, 5)
    )  # Adjusted for three metrics (Reward, Total Reward, Stockouts)

    # Plot the 'Reward' over time
    # for i, (df, label) in enumerate(zip(dataframes, labels)):
    #     axs[0].plot(
    #         df["Time"], df["Reward"], label=f"{label} - Reward", color=colors[i]
    #     )
    # axs[0].set_title("Reward over Time")
    # axs[0].legend()

    for i, (df, label) in enumerate(zip(dataframes, labels)):
        smoothed_reward = smooth(df["Reward"], window_size=80)
        axs[0].plot(
            df["Time"], smoothed_reward, label=f"{label} - Reward", color=colors[i]
        )
    axs[0].set_title("Reward over Time")
    axs[0].legend()

    # Plot the 'Total Reward' over time
    for i, (df, label) in enumerate(zip(dataframes, labels)):
        axs[1].plot(
            df["Time"],
            df["Total Reward"],
            label=f"{label} - Total Reward",
            color=colors[i],
        )
    axs[1].set_title("Total Reward over Time")
    axs[1].legend()
    # axs[1].set_yscale("log")

    # Plot stockouts (backlog) over time
    for i, (df, label) in enumerate(zip(dataframes, labels)):
        stockouts = df["Backlog"].sum()
        axs[2].plot(
            df["Time"],
            df["Backlog"],
            label=f"{label} - Stockouts ({stockouts})",
            color=colors[i],
        )
    axs[2].set_title("Stockouts Over Time")
    axs[2].legend()

    # Adjust spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    # Show the plot
    plt.show()


# Define a smoothing function for improved visualization and interpretation
def smooth(data, window_size):
    return data.rolling(window=window_size, center=True).mean()
