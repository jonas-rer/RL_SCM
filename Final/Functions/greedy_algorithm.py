import numpy as np


# def greedy_algorithm(env, num_episodes, episode_length):
#     total_rewards = []

#     for episode in range(num_episodes):
#         state, info = env.reset()  # Unpack the reset tuple

#         episode_reward = 0
#         done = False

#         for step in range(episode_length):
#             # Save the current state to restore later
#             saved_state = env.save_state()

#             rewards_for_actions = []
#             actions = []

#             # Generate all possible actions for the multidiscrete action space
#             for action in np.ndindex(*env.action_space.nvec):
#                 env.load_state(saved_state)  # Restore the environment state
#                 _, reward, _, _, _ = env.step(action)  # Simulate the action
#                 rewards_for_actions.append(reward)
#                 actions.append(action)

#             future_rewards = [
#                 lookahead(env, state, depth=3)
#                 for action in np.ndindex(*env.action_space.nvec)
#             ]

#             # Choose the action with the highest immediate reward
#             # best_action_index = np.argmax(rewards_for_actions)
#             best_action_index = np.argmax(future_rewards)
#             # best_action_index = np.argmax(mc_rewards)
#             best_action = actions[best_action_index]

#             # Take the best action in the main environment
#             state, reward, done, _, _ = env.step(best_action)
#             episode_reward += reward

#             if done:
#                 break

#             # env.render()

#         total_rewards.append(episode_reward)

#     avg_reward = sum(total_rewards) / num_episodes
#     print(f"Average reward over {num_episodes} episodes: {avg_reward}")

#     return total_rewards


def greedy_algorithm(
    env, num_episodes, episode_length, lookahead_depth=3, discount_factor=0.9
):
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0

        for step in range(episode_length):
            saved_state = env.save_state()
            best_action = None
            best_score = float("-inf")

            for action in np.ndindex(*env.action_space.nvec):
                env.load_state(saved_state)
                _, immediate_reward, done, _, _ = env.step(action)

                if done or env.episode_length <= 0:
                    total_score = immediate_reward
                else:
                    future_reward = lookahead(
                        env, state, lookahead_depth, discount_factor
                    )
                    total_score = immediate_reward + discount_factor * future_reward

                if total_score > best_score:
                    best_score = total_score
                    best_action = action

            # Take the best action in the main environment
            state, reward, done, _, _ = env.step(best_action)
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    return total_rewards


def lookahead(env, state, depth, discount_factor=0.9):
    """
    Lookahead function that simulates the environment for a given number of steps
    and returns the total reward for the best action.
    """

    if depth == 0:
        return 0  # Base case

    saved_state = env.save_state()  # Save current state
    best_reward = float("-inf")

    for action in np.ndindex(*env.action_space.nvec):
        env.load_state(saved_state)  # Restore state for each action
        _, reward, done, _, _ = env.step(action)
        if done:
            best_reward = max(best_reward, reward)
        else:
            future_reward = lookahead(env, state, depth - 1)
            total_reward = reward + discount_factor * future_reward
            best_reward = max(best_reward, total_reward)

    return best_reward


# TODO - Implement Monte Carlo simulation

# mc_rewards = [
#     monte_carlo_simulation(env, state, action)
#     for action in np.ndindex(*env.action_space.nvec)
# ]


def monte_carlo_simulation(env, state, action, num_rollouts=10, rollout_length=5):
    """
    Monte Carlo simulation to estimate the expected reward for a given action.
    """
    total_reward = 0
    saved_state = env.save_state()

    for _ in range(num_rollouts):
        env.load_state(saved_state)
        env.step(action)  # Take the initial action
        reward = 0
        for _ in range(rollout_length):
            random_action = tuple(env.action_space.sample())
            _, r, done, _, _ = env.step(random_action)
            reward += r
            if done:
                break
        total_reward += reward

    return total_reward / num_rollouts  # Average reward
