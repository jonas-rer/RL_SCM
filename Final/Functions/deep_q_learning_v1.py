import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import deque
from tensorflow.keras.callbacks import TensorBoard


class DQNTrainer:
    def __init__(
        self,
        env,
        learning_rate=0.001,
        discount_rate=0.99,
        exploration_rate=1.0,
        max_exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.001,
        batch_size=64,
        memory_size=10000,
        tensorboard_log_path="./logs",
    ):
        self.env = env

        # Extract observation and action space details
        sample_observation = env.reset()
        self.state_size = len(self.flatten_observation(sample_observation))
        self.action_sizes = env.action_space.nvec  # For MultiDiscrete action spaces

        # Create the Q-network
        self.q_network = self.create_q_network(
            self.state_size, self.action_sizes, learning_rate
        )

        # Parameters
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # TensorBoard setup
        self.tensorboard_log_path = tensorboard_log_path
        self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_log_path)

        # Training metrics
        self.rewards_all_episodes = []

    @staticmethod
    def flatten_observation(observation):
        """Flattens the Dict observation into a single vector."""
        return np.concatenate(
            [
                observation["inventory_levels"].flatten(),  # Ensure 1D
                observation["current_demand"].flatten(),  # Ensure 1D
                observation["backlog_levels"].flatten(),  # Ensure 1D
                observation["order_queues"].flatten(),  # Ensure 1D
                observation["lead_times"].flatten(),  # Ensure 1D
            ]
        )

    @staticmethod
    def create_q_network(state_size, action_sizes, learning_rate=0.001):
        """Creates a Q-network with multiple outputs for MultiDiscrete action spaces."""
        inputs = layers.Input(shape=(state_size,))
        common_layer = layers.Dense(64, activation="relu")(inputs)
        common_layer = layers.Dense(64, activation="relu")(common_layer)

        outputs = [
            layers.Dense(action_size, activation="linear")(common_layer)
            for action_size in action_sizes
        ]
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
        )
        return model

    def train(self, total_timesteps):
        """Trains the Q-network for the specified number of timesteps."""
        current_timestep = 0
        episode = 0

        while current_timestep < total_timesteps:
            state = self.env.reset()  # Reset the environment
            state = self.flatten_observation(state)
            state = np.reshape(state, [1, self.state_size])  # Reshape for Q-network

            done = False
            rewards_current_episode = 0

            while not done and current_timestep < total_timesteps:
                # Exploration-exploitation trade-off
                exploration_rate_threshold = np.random.uniform(0, 1)
                if exploration_rate_threshold > self.exploration_rate:
                    # Random action for MultiDiscrete: sample each sub-action independently
                    action = [
                        np.random.choice(action_size)
                        for action_size in self.action_sizes
                    ]
                else:
                    # Predicted action: Choose the best action for each sub-action
                    q_values = self.q_network.predict(state)
                    action = [np.argmax(q) for q in q_values]

                # Convert the action to a NumPy array (if needed by the environment)
                action = np.array(action)
                actions = np.array([action] * self.env.num_envs)

                # Take action
                new_state, reward, done, truncated = self.env.step(actions)
                new_state = self.flatten_observation(new_state)
                new_state = np.reshape(new_state, [1, self.state_size])

                # Store experience in memory
                self.memory.append((state, action, reward, new_state, done))
                state = new_state
                rewards_current_episode += reward

                # Train the Q-network with a mini-batch
                if len(self.memory) >= self.batch_size:
                    self.train_step()

                current_timestep += 1

            # Decay exploration rate
            self.exploration_rate = self.min_exploration_rate + (
                self.max_exploration_rate - self.min_exploration_rate
            ) * np.exp(-self.exploration_decay_rate * episode)

            # Log episode metrics
            self.rewards_all_episodes.append(rewards_current_episode)
            self.log_metrics(episode, rewards_current_episode)

            episode += 1

        # Save the trained model
        self.q_network.save(os.path.join(self.tensorboard_log_path, "q_network.h5"))

    def train_step(self):
        """Performs one training step using a mini-batch from memory."""
        # Ensure enough data in replay buffer
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to numpy arrays
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=int)

        # Normalize states
        states = self.normalize(states)
        next_states = self.normalize(next_states)

        # Compute Q-values for current and next states
        q_values = self.q_network.predict(states)
        q_values_next = self.q_network.predict(next_states)

        # Compute target Q-values
        targets = [q.copy() for q in q_values]
        for i in range(self.batch_size):
            for j in range(
                len(self.action_sizes)
            ):  # Iterate over MultiDiscrete components
                target_q_value = rewards[i]
                if not dones[i]:
                    target_q_value += self.discount_rate * np.max(q_values_next[j][i])
                targets[j][i, actions[i][j]] = target_q_value

        # Train the Q-network on the batch for multiple epochs
        self.q_network.fit(
            states, targets, epochs=5, batch_size=self.batch_size, verbose=1
        )

    def log_metrics(self, episode, reward):
        """Logs metrics to TensorBoard."""
        with self.tensorboard_writer.as_default():
            tf.summary.scalar("Reward", reward, step=episode)
            tf.summary.scalar("Exploration Rate", self.exploration_rate, step=episode)
            tf.summary.scalar(
                "Average Reward (Last 10)",
                np.mean(self.rewards_all_episodes[-10:]),
                step=episode,
            )
            self.tensorboard_writer.flush()
