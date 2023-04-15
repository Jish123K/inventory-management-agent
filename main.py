import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow_probability import distributions as tfd

class InventoryManager:

    def __init__(self, env, policy, gamma=0.99):

        self.env = env

        self.policy = policy

        self.gamma = gamma

    def run(self, num_episodes=1000):

        rewards = []

        for episode in range(num_episodes):

            # Initialize state

            state = self.env.reset()

            # Initialize action

            action = self.policy(state)

            # Take action and observe reward and next state

            next_state, reward, done, _ = self.env.step(action)

            # Update rewards

            rewards.append(reward)

            # Update state

            state = next_state

            # If done, break episode

            if done:

                break

        return rewards

class Policy:

    def __init__(self, env, model):

        self.env = env

        self.model = model
        def __call__(self, state):

        # Get probabilities of taking each action

        probs = self.model.predict(state)

        # Choose action according to probabilities

        action = np.argmax(probs)

        return action

class Environment:

    def __init__(self, data, num_items, num_periods):

        self.data = data

        self.num_items = num_items

        self.num_periods = num_periods

    def reset(self):

        self.inventory = np.zeros(self.num_items)

        self.demand = np.zeros(self.num_periods)

        self.sales = np.zeros(self.num_periods)

        self.cost = 0

        return self.inventory

    def step(self, action):

        # Order the specified number of items

        self.inventory += action

        # Generate demand

        self.demand = np.random.poisson(self.data['demand'] / self.data['price'])

        # Sell as many items as possible

        self.sales = min(self.inventory, self.demand)

        # Calculate cost

        self.cost += self.data['cost'] * self.sales + self.data['storage'] * self.inventory

        # Return next state, reward, and done

        return self.inventory, self.sales - self.cost, self.demand == 0, {}

def main():

    # Load data

    data = pd.read_csv('data/inventory.csv')

    # Create environment

    env = Environment(data, 10, 100)

    # Create policy
        policy = Policy(env, Sequential([

        Dense(128, activation='relu'),

        Dense(64, activation='relu'),

        Dense(env.num_items, activation='softmax')

    ]))

    # Train policy

    model = policy.model

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    callbacks = [EarlyStopping(patience=10)]

    model.fit(env.reset(), env.run(num_episodes=10000), epochs=100, callbacks=callbacks)

    # Evaluate policy

    rewards = InventoryManager(env, policy).run(num_episodes=1000)

    print('Average reward:', np.mean(rewards))
    # Add more functionality and not repeat previous code in any condition

# 1. Add a function to plot the rewards over time.

def plot_rewards(rewards):

    plt.plot(rewards)

    plt.xlabel('Episode')

    plt.ylabel('Reward')

    plt.show()

# 2. Add a function to save the trained policy.

def save_policy(policy):

    policy.model.save('policy.h5')

# 3. Add a function to load a saved policy.

def load_policy():

    policy = Policy(env, Sequential())

    policy.model = tf.keras.models.load_model('policy.h5')

    return policy

# 4. Add a function to visualize the policy.

def visualize_policy(policy):

    for state in env.states:

        probs = policy(state)

        plt.bar(range(env.num_items), probs)

        plt.xlabel('Item')

        plt.ylabel('Probability')

        plt.show()

# 5. Add a function to test the policy.

def test_policy(policy):

    rewards = []

    for episode in range(100):

        state = env.reset()

        done = False

        while not done:

            action = policy(state)

            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)

            state = next_state

    print('Average reward:', np.mean(rewards))
    # 6. Add a function to optimize the policy.

def optimize_policy(policy, env, num_iterations=10000):

    for iteration in range(num_iterations):

        # Generate a batch of episodes.

        episodes = env.generate_episodes(policy)

        # Update the policy.

        policy.model.fit(episodes, epochs=1)

# 7. Add a function to explore the environment.

def explore_environment(env, num_steps=1000):

    for step in range(num_steps):

        # Generate a random action.

        action = np.random.randint(env.num_items)

        # Take the action and observe the reward and next state.

        next_state, reward, done, _ = env.step(action)

        # Update the environment.

        env.update(action, reward, next_state, done)

# 8. Add a function to visualize the environment.

def visualize_environment(env):

    for state in env.states:

        plt.scatter(state[0], state[1], c='red')

    plt.show()
    # 9. Add a function to generate a random policy.

def generate_random_policy(env):

    policy = {}

    for state in env.states:

        policy[state] = np.random.randint(env.num_items)

    return policy

# 10. Add a function to find the optimal policy using value iteration.

def value_iteration(env, gamma=0.99):

    # Initialize value function.

    v = np.zeros(env.num_states)

    # Iterate until convergence.

    while True:

        # Update value function.

        for state in env.states:

            v[state] = np.max([gamma * np.sum([p * v[next_state] for p, next_state in env.transitions[state].items()]) for p in env.rewards[state]])

        # Check for convergence.

        delta = np.max(np.abs(v - np.roll(v, 1)))

        if delta < 1e-6:

            break

    return v

# 11. Add a function to find the optimal policy using policy iteration.

def policy_iteration(env, gamma=0.99):

    # Initialize policy.

    policy = generate_random_policy(env)

    # Iterate until convergence.

    while True:

        # Evaluate policy.

        value_function = np.zeros(env.num_states)

        for state in env.states:

            for action in env.actions[state]:
                value_function[state] = max([gamma * np.sum([p * v[next_state] for p, next_state in env.transitions[state][action].items()]) for p in env.rewards[state][action]
                                             )
                                             # Check for convergence.

        if policy == env.policy:

            break

    return policy

# 12. Add a function to compare different policies.

def compare_policies(env, policies):

    rewards = []

    for policy in policies:

        rewards.append(evaluate_policy(env, policy))

    plt.plot(rewards)

    plt.xlabel('Policy')

    plt.ylabel('Reward')

    plt.show()
                                             

# 13. Add a function to train a policy using Q-learning.

def q_learning(env, gamma=0.99, alpha=0.01, epsilon=0.1, num_episodes=10000):

    # Initialize Q-table.

    Q = np.zeros((env.num_states, env.num_actions))

    # Iterate over episodes.

    for episode in range(num_episodes):

        # Initialize state.

        state = env.reset()

        # Iterate over steps.

        for step in range(env.max_steps):

            # Choose action.

            if np.random.random() < epsilon:

                action = np.random.randint(env.num_actions)

            else:

                action = np.argmax(Q[state])

            # Take action and observe reward and next state.

            next_state, reward, done, _ = env.step(action)

            # Update Q-table.

            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # Update state.

            state = next_state

            # If done, break episode.

            if done:

                break

    return Q

# 14. Add a function to train a policy using SARSA.

def sarsa(env, gamma=0.99, alpha=0.01, epsilon=0.1, num_episodes=10000):

    # Initialize Q-table.

    Q = np.zeros((env.num_states, env.num_actions))

    # Iterate over episodes.

    for episode in range(num_episodes):

        # Initialize state and action.

        state = env.reset()

        action = np.random.randint(env.num_actions)
                                             # Iterate over steps.

        for step in range(env.max_steps):

            # Take action and observe reward and next state.

            next_state, reward, done, _ = env.step(action)

            # Update Q-table.

            Q[state][action] += alpha * (reward + gamma * Q[next_state][np.argmax(Q[next_state])] - Q[state][action])

            # Update state and action.

            state = next_state

            action = np.argmax(Q[state])

            # If done, break episode.

            if done:

                break

    return Q

# 15. Add a function to train a policy using Deep Q-learning.

def dqn(env, gamma=0.99, alpha=0.001, epsilon=0.1, num_episodes=10000, batch_size=32, memory_size=10000):

    # Initialize DQN.

    model = Sequential([

        Dense(128, activation='relu'),

        Dense(64, activation='relu'),

        Dense(env.num_actions, activation='softmax')

    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Initialize memory.

    memory = ReplayMemory(memory_size)

    # Iterate over episodes.

    for episode in range(num_episodes):

        # Initialize state.

        state = env.reset()

        # Iterate over steps.

        for step in range(env.max_steps):

            # Choose action.

            if np.random.random() < epsilon:

                action = np.random.randint(env.num_actions)

            else:
                                             # Store experience in memory.

            memory.add((state, action, reward, next_state, done))

            # If enough experience is stored, train the DQN.

            if len(memory) >= batch_size:

                # Sample a batch of experiences from memory.

                experiences = memory.sample(batch_size)

                # Train the DQN on the batch of experiences.

                loss = model.train_on_batch(experiences)

            # Update epsilon.

            epsilon *= 0.995

            # Update state.

            state = next_state

            # If done, break episode.

            if done:

                break

    return model

def main():

    # Load data.

    data = pd.read_csv('data/inventory.csv')

    # Create environment.

    env = Environment(data, 10, 100)

    # Create policy.

    policy = generate_random_policy(env)

    # Train policy using Q-learning.

    Q = q_learning(env, gamma=0.99, alpha=0.01, epsilon=0.1, num_episodes=10000)

    # Evaluate policy.

    rewards = evaluate_policy(env, policy)

    print('Average reward:', np.mean(rewards))

    # Train policy using SARSA.

    Q = sarsa(env, gamma=0.99, alpha=0.01, epsilon=0.1, num_episodes=10000)
                                             

    # Evaluate policy.

    rewards = evaluate_policy(env, policy)

    print('Average reward:', np.mean(rewards))

if __name__ == '__main__':

    main()
                                             print('Done')
