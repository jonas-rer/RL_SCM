import numpy as np


def greedy_algorithm(env, num_episodes, episode_length):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle tuple return type if necessary
            state = state[0]

        episode_reward = 0
        done = False

        for step in range(episode_length):
            # Evaluate the immediate reward for each action in the current state
            rewards_for_actions = []
            actions = []

            # Generate all possible actions for the multidiscrete action space
            for action in np.ndindex(*env.action_space.nvec):
                # Use a copy of the environment to estimate rewards without affecting the main env
                temp_env = env.clone() if hasattr(env, "clone") else env
                temp_env.reset()
                temp_state, reward = temp_env.step(action)[:2]
                rewards_for_actions.append(reward)
                actions.append(action)

            # Choose the action with the highest immediate reward
            best_action_index = np.argmax(rewards_for_actions)
            best_action = actions[best_action_index]
            state, reward, done = env.step(best_action)[
                :3
            ]  # Unpack only the first three values

            episode_reward += reward

            if done:
                break

            env.render()

        total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    return total_rewards
