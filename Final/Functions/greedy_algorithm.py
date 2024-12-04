import numpy as np


def greedy_algorithm(env, num_episodes, episode_length):
    """
    Greedy algorithm that selects the best action based on immediate rewards.
    """

    total_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0

        try:
            for step in range(episode_length):
                best_action = None
                best_score = float("-inf")

                # Evaluate all possible actions
                for action in np.ndindex(*env.action_space.nvec):
                    env.load_state(
                        env.save_state()
                    )  # Save and restore state for evaluation

                    try:
                        _, immediate_reward, done, _, _ = env.step(action)
                    except IndexError:
                        # Handle invalid timestep error
                        immediate_reward = -float("inf")  # Penalize invalid actions
                        done = True  # End evaluation for this action

                    if immediate_reward > best_score:
                        best_score = immediate_reward
                        best_action = action

                # Take the best action in the main environment
                state, reward, done, _, _ = env.step(best_action)
                episode_reward += reward

                if done:
                    break

        except IndexError as e:
            # Handle unexpected environment errors
            print(f"Encountered IndexError: {e}")
            break

        total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    return total_rewards


def greedy_algorithm_lookahead(
    env, num_episodes, episode_length, lookahead_depth=3, discount_factor=0.9
):
    """
    Greedy algorithm with lookahead to evaluate future rewards.
    """

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


def monte_carlo_agent(
    env,
    num_episodes,
    episode_length,
    lookahead_depth=3,
    discount_factor=0.9,
    num_rollouts=10,
    rollout_length=5,
):
    """
    Run an agent in the environment using Monte Carlo lookahead for decision-making.
    """
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

                if done:
                    future_reward = immediate_reward
                else:
                    future_reward = lookahead_with_monte_carlo(
                        env,
                        state,
                        lookahead_depth,
                        discount_factor,
                        num_rollouts,
                        rollout_length,
                    )
                    total_score = immediate_reward + discount_factor * future_reward

                if total_score > best_score:
                    best_score = total_score
                    best_action = action

            # Take the best action in the actual environment
            state, reward, done, _, _ = env.step(best_action)
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} reward: {episode_reward}")

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    return total_rewards


def lookahead_with_monte_carlo(
    env, state, depth, discount_factor, num_rollouts=10, rollout_length=5
):
    """
    Lookahead function using Monte Carlo simulations to estimate future rewards.
    """
    if depth == 0:
        return 0  # No future reward beyond the depth

    saved_state = env.save_state()
    best_future_reward = float("-inf")

    for action in np.ndindex(*env.action_space.nvec):
        env.load_state(saved_state)
        _, immediate_reward, done, _, _ = env.step(action)

        if done:
            future_reward = immediate_reward
        else:
            future_reward = monte_carlo_simulation(
                env, state, action, num_rollouts, rollout_length
            )
            future_reward += discount_factor * lookahead_with_monte_carlo(
                env, state, depth - 1, discount_factor, num_rollouts, rollout_length
            )

        best_future_reward = max(best_future_reward, future_reward)

    return best_future_reward


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
