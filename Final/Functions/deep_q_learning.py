# %%
import numpy as np
import tensorflow as tf
from keras import layers, models
import random
from collections import deque

# %%


# Define the Q-network
def create_q_network(state_size, action_size, learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(state_size,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(action_size, activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model


# Deep Q-learning with experience replay
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
):
    state_size = env.observation_space.shape[0]  # For continuous state spaces
    action_size = env.action_space.n

    # Initialize Q-network
    q_network = create_q_network(state_size, action_size, learning_rate)

    # Replay memory
    memory = deque(maxlen=memory_size)
    rewards_all_episodes = []

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
                action = np.argmax(q_network.predict(state))  # Exploitation
            else:
                action = env.action_space.sample()  # Exploration

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
                targets = q_values.copy()
                for i in range(batch_size):
                    target_q_value = rewards[i]
                    if not dones[i]:
                        target_q_value += discount_rate * np.max(q_values_next[i])
                    targets[i, actions[i]] = target_q_value

                # Train the Q-network on the batch
                q_network.fit(states, targets, epochs=1, verbose=0)

            if done:
                break

        # Decay exploration rate
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * episode)

        rewards_all_episodes.append(rewards_current_episode)

    return rewards_all_episodes, q_network


# %%
