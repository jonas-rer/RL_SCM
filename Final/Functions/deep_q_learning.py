import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import deque


def create_q_network(state_size, action_sizes, learning_rate=0.001):
    # Use Keras functional API for multi-output model
    inputs = layers.Input(shape=(state_size,))
    common_layer = layers.Dense(64, activation="relu")(inputs)
    common_layer = layers.Dense(64, activation="relu")(common_layer)

    # Create a separate output for each discrete action component
    outputs = [
        layers.Dense(action_size, activation="linear")(common_layer)
        for action_size in action_sizes
    ]
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model


def deep_q_learning(
    env,
    num_episodes,
    max_steps_per_episode,
    learning_rate,
    discount_rate,
    exploration_rate,
    max_exploration_rate,
    min_exploration_rate,
    exploration_decay_rate,
    batch_size=64,
    memory_size=10000,
    model_save_path="q_network_model",
):
    state_size = env.observation_space.shape[0]  # For continuous state spaces
    action_sizes = env.action_space.nvec

    q_network = create_q_network(state_size, action_sizes, learning_rate)
    memory = deque(maxlen=memory_size)
    rewards_all_episodes = []
    exploration_rates = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle tuple return type if necessary
            state = state[0]
        state = np.reshape(state, [1, state_size])  # Reshape for network input

        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = np.random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                q_values = q_network.predict(state)
                action = [
                    np.argmax(q) for q in q_values
                ]  # Choose the best action for each part
            else:
                action = [
                    np.random.choice(action_size) for action_size in action_sizes
                ]  # Random action for each part

            # Take action and observe the new state and reward
            new_state, reward, done, *_ = env.step(action)
            new_state = np.reshape(new_state, [1, state_size])

            # Store experience in memory
            memory.append((state, action, reward, new_state, done))
            state = new_state
            rewards_current_episode += reward

            # Sample mini-batch from memory and perform gradient descent
            if len(memory) >= batch_size:
                mini_batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*mini_batch)

                # Convert to numpy arrays
                states = np.vstack(states)
                next_states = np.vstack(next_states)
                rewards = np.array(rewards)
                dones = np.array(dones, dtype=int)

                # Compute Q-values for current and next states
                q_values = q_network.predict(states)
                q_values_next = q_network.predict(next_states)

                # Compute target Q-values
                targets = [q.copy() for q in q_values]
                for i in range(batch_size):
                    for j in range(
                        len(action_sizes)
                    ):  # Iterate over each component in MultiDiscrete space
                        target_q_value = rewards[i]
                        if not dones[i]:
                            target_q_value += discount_rate * np.max(
                                q_values_next[j][i]
                            )
                        targets[j][
                            i, actions[i][j]
                        ] = target_q_value  # Update the specific component

                # Train the Q-network on the batch
                q_network.fit(states, targets, epochs=1, verbose=0)

            if done:
                break

        # Decay exploration rate
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * episode)

        # Append metrics
        rewards_all_episodes.append(rewards_current_episode)
        exploration_rates.append(exploration_rate)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            average_reward = np.mean(rewards_all_episodes[-10:])
            print(
                f"Episode: {episode + 1}, "
                f"Reward: {rewards_current_episode}, "
                f"Average Reward (last 10): {average_reward:.2f}, "
                f"Exploration Rate: {exploration_rate:.4f}"
            )

    # Save the trained model
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    q_network.save(os.path.join(model_save_path, "q_network.h5"))

    return rewards_all_episodes, exploration_rates, q_network
