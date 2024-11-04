import numpy as np


def q_learning(
    env,
    num_episodes,
    max_steps_per_episode,
    learning_rate,
    discount_rate,
    exploration_rate,
    max_exploration_rate,
    min_exploration_rate,
    exploration_decay_rate,
):
    # Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))

    # Q-learning algorithm
    rewards_all_episodes = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = np.random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            # Update Q-table for Q(s,a)
            q_table[state, action] = q_table[state, action] * (
                1 - learning_rate
            ) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done == True:
                break

        # Exploration rate decay
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * episode)

        rewards_all_episodes.append(rewards_current_episode)

    return rewards_all_episodes, q_table
