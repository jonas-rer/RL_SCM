import numpy as np
import gym  # or your specific environment package


def greedy_algorithm(env, num_episodes, max_steps_per_episode):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle tuple return type if necessary
            state = state[0]

        episode_reward = 0
        done = False

        for step in range(max_steps_per_episode):
            # Evaluate the immediate reward for each action in the current state
            rewards_for_actions = []
            for action in range(env.action_space.n):
                # Take the action temporarily to estimate reward (requires env to support this)
                temp_state, reward, _, _ = env.step(action)
                rewards_for_actions.append(reward)
                env.reset()  # Reset to initial state to not alter environment

            # Choose the action with the highest immediate reward
            best_action = np.argmax(rewards_for_actions)
            state, reward, done, _ = env.step(best_action)

            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    return total_rewards
