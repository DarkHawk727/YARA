import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt

class DummyEnv:
    def __init__(self):
        self.state_size = 100  # Assume a large state space
        self.action_size = 2
        self.max_steps = 1

    def reset(self):
        self.current_step = 0
        return np.random.rand(self.state_size)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False

        next_state = np.random.rand(self.state_size)
        reward = -100*np.random.rand(1)[0] # reward random asign value between 0 and -100 

        return next_state, reward, done, {}

    


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
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))  # Increased units in the input layer
        model.add(Dense(128, activation='relu'))  # Added another hidden layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self): # offline model check on online model once a while to update weights on its neurons
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
env = DummyEnv()
state_size = env.state_size
action_size = env.action_size

agent = DQNAgent(state_size, action_size)

# Lists to store episode rewards
episode_rewards = []

for episode in range(100):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0  # Track total reward for this episode

    for time in range(50):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        total_reward += reward  # Accumulate the reward for this episode
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            break

    episode_rewards.append(total_reward)

# Plotting
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Learning Progress')
plt.show()