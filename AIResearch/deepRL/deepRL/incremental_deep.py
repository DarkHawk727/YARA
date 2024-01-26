import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Define a simple Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def update_q_network(self, batch):
        states, actions, rewards, next_states, dones = batch

        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1).values

        predicted_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define and train the DQN agent
def train_dqn(env, agent, episodes, max_steps, batch_size, target_update_frequency):
    for episode in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            agent.update_q_network((state, action, reward, next_state, done))

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decrease_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Define a replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define a DQN agent that supports experience replay
class DQNAgentWithExperienceReplay:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=10000)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def update_q_network(self, batch):
        states, actions, rewards, next_states, dones = batch

        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1).values

        predicted_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def load_experience(self, buffer):
        self.replay_buffer = buffer

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

# Define the custom environment
class CustomEnvironment:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.random.rand(state_size)
        self.solution = np.random.rand(action_size)
        self.done = False

    def reset(self):
        self.state = np.random.rand(self.state_size)
        self.done = False
        return self.state

    def step(self, action):
        # Calculate Kendall tau distance as reward
        kendall_tau, _ = kendalltau(action, self.solution)
        reward = -kendall_tau  # Negative Kendall tau as a reward
        self.done = True
        return self.state, reward, self.done

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define and train the DQN agent
# Define and train the DQN agent
def train_dqn(agent, env, episodes, max_steps, batch_size, target_update_frequency):
    replay_buffer = ReplayBuffer(buffer_size=10000)

    rewards = []  # To store rewards for plotting

    for episode in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            replay_buffer.add(state, action, reward, next_state, done)

            if len(replay_buffer.buffer) > batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                agent.update_q_network(batch)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decrease_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        rewards.append(total_reward)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # Plot learning outcome
    plt.plot(range(episodes), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Outcome')
    plt.show()


# Define the custom environment
class CustomEnvironment:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.random.rand(state_size)
        self.solution = np.random.rand(action_size)
        self.done = False

    def reset(self):
        self.state = np.random.rand(self.state_size)
        self.done = False
        return self.state

    def step(self, action):
        # Calculate Kendall tau distance as reward
        kendall_tau, _ = kendalltau(action, self.solution)
        reward = -kendall_tau  # Negative Kendall tau as a reward
        self.done = True
        return self.state, reward, self.done

# Create and train the DQN agent
if __name__ == "__main__":
    state_size = 128
    action_size = 5

    agent = DQNAgent(state_size, action_size)
    env = CustomEnvironment(state_size, action_size)

    # Train the agent
    train_dqn(agent, env, episodes=100, max_steps=100, batch_size=32, target_update_frequency=10)

# Create and train the DQN agent
if __name__ == "__main__":
    state_size = 128
    action_size = 5

    agent = DQNAgentWithExperienceReplay(state_size, action_size)

    # Load previous experiences and model (if available)
    try:
        agent.load_experience(torch.load("experience_buffer.pt"))
        agent.load_model("dqn_model.pt")
        print("Loaded previous experiences and model.")
    except FileNotFoundError:
        print("No previous experiences and model found.")

    env = CustomEnvironment(state_size, action_size)

    # Train the agent
    train_dqn(agent, env, episodes=100, max_steps=100, batch_size=32, target_update_frequency=10)

    # Save experiences and model for future use
    torch.save(agent.replay_buffer, "experience_buffer.pt")
    agent.save_model("dqn_model.pt")