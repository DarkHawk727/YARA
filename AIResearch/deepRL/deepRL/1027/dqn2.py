"""
The agent just going to pull up a ranking that it sees as a most likely solution at 
given state and check to see what reward it receives, and through interacting and 
iteration of possibly the same state at some point in time while trying different 
actions, the agent collectively learns the best solution by adjusting the neural 
network's neuron weight.

Using continuous state of vector size m 
Action: start with size n! doc chunk search, them cool down to k=1, adaptvely cools to prevent 
over-exploration.
 agent use k choose n, top k response selected ordered, P(10,3)

"""

import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DummyEnv:
    def __init__(self, n):
        self.state_size = 6  # State represents the agent's guess for the top k ranking of n numbers
        self.action_size = 1  # Agent can choose to adjust its guess
        self.max_steps = 1

    def reset(self):
        return np.random.rand(self.state_size)

    def step(self, action):
        # For simplicity, assume action is 1 if agent chooses to adjust guess, else 0
        if action == 1:
            done = True
            # Provide the correct ranking as feedback
            correct_ranking = np.arange(1, self.state_size+1)
        else:
            done = False
            correct_ranking = np.zeros(self.state_size)  # Placeholder for no adjustment

        reward = 0  # Reward can be computed based on the similarity between agent's guess and correct ranking

        return correct_ranking, reward, done



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            a = self.model.predict(next_state)[0]
            t = self.target_model.predict(next_state)[0]
            target[0][action] = reward + self.gamma * t[np.argmax(a)]
        self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


# Example usage
n = 10  # Number of items to rank
env = DummyEnv(n)
state_size = env.state_size
action_size = env.action_size

agent = DQNAgent(state_size, action_size)

# Lists to store episode rewards
episode_rewards = []

for episode in range(10):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0  # Track total reward for this episode

    for time in range(5):
        action = agent.act(state)
        correct_ranking, reward, done = env.step(action)
        total_reward += reward  # Accumulate the reward for this episode
        next_state = np.reshape(correct_ranking, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    episode_rewards.append(total_reward)

# Plotting
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Learning Progress')
plt.show()

