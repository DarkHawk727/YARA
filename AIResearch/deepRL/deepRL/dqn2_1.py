"""
The agent just going to pull up a ranking that it sees as a most likely solution at 
given state and check to see what reward it receives, and through interacting and 
iteration of possibly the same state at some point in time while trying different 
actions, the agent collectively learns the best solution by adjusting the neural 
network's neuron weight.

Using continuous state of vector size m 
Action: start with size n! doc chunk search, them cool down to k=1, adaptvely cools to prevent 
over-exploration. Kendall tau distance
 agent use k choose n, top k response selected ordered, P(10,3)

 key nots:
 1.Use Masking for padding in vector size inconsistencies.
 2.state size differs, as they are keyword embeddings.
 3.using recurrent layers (like LSTM or GRU) to adopt changing state len.
 4. env reward and agent update: Using a combination of ground truth training and test sets 
 5. see ### for original implemntation for conventional DQN and another ### with reasoning 

 

"""

import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy.stats import kendalltau
from tensorflow.keras.layers import LSTM #recurrent layer
import csv # state | best action learned

class DummyEnv: 
    def __init__(self, n, k, noise_level=0.8):
        self.state_size = n # dimension of state
        self.action_size = 1
        self.max_steps = 1
        self.k = k
        self.correct_ranking = np.arange(1, n+1)
        self.noise_level = noise_level

    def reset(self):
        self.current_step = 0
        self.current_state = np.random.choice(self.correct_ranking, size=self.k, replace=False)

        # Add random noise to the initial state
        self.current_state = self.current_state.astype(np.float32)
        self.current_state += np.random.normal(0, self.noise_level, size=self.k)
        
        return self.current_state


    def step(self, action):
        done = True
        # Calculate Kendall tau distance and convert it to a reward (higher similarity -> higher reward)
        kendall_distance = abs(kendalltau(self.current_state, self.correct_ranking[:self.k])[0]) #
        return self.current_state, kendall_distance, done

    ### Use for Ground truth case: reasoning- agent knowing the correct ranking from ground truth, to save time exploring various ranking,
         #we selectively upate the correct ranking only
    def get_correct_ranking(self):
        return self.correct_ranking  # Return the correct ranking





class RecurrentDQNAgent:
    def __init__(self, state_size, action_size, ground_truth=False):
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
        self.state_to_best_action = {}  # Dictionary to store best actions
        
        ### Ground truth: GPT or Developer prompted trainset with solutions 
        self.ground_truth=ground_truth 

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, self.state_size), activation='relu'))
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
        
        # Convert state to a string for use as a dictionary key
        state_str = ','.join(map(str, state[0]))
        
        # Update state-to-best-action mapping
        if state_str not in self.state_to_best_action or target[0][action] > self.state_to_best_action[state_str]:
            self.state_to_best_action[state_str] = target[0][action]

        self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    def save_state_action_mappings(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['State', 'Best Action'])  # Write header row

            for state, best_action in self.state_to_best_action.items():
                writer.writerow([','.join(map(str, state)), best_action])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


# Example usage

# Example usage
n = 10  # total action
k = 3   # top k choice
env = DummyEnv(n, k)

state_size = 300  # Dimensionality of the word embeddings
action_size = n   # Total number of possible actions
agent = RecurrentDQNAgent(state_size, action_size)


# Lists to store episode rewards
episode_rewards = []

for episode in range(100):
    state = env.reset() #state episdoe to start with #### replaced by keyword embedding 
    padded_state = np.zeros((1, state_size, state_size))  # Initialize with zeros
    padded_state[0, :len(state), :len(state)] = state 
    state = padded_state
    total_reward = 0  # Track total reward for this episode
    ### our case : reasoning- reduce env interaction, just adjust gradient on the network 
    action = agent.act(state)
    correct_ranking, reward, done = env.step(action)  ### we store result here for recap, env has GPT API call, we minimize GPT API calls 

    ###

    for time in range(5): # time steps within each episode means number of times agent interact with env
    ### original implemntation
        #action = agent.act(state) ###
        #correct_ranking, reward, done = env.step(action) ### standard DQN, but we assume this loop is used to recap learning only (i.e., agent update only, no env ) see 
        
    ### 
        total_reward += reward  # Accumulate the reward for this episode
        padded_state = np.zeros(state_size)
        padded_state[:len(correct_ranking)] = correct_ranking #let padding 0, avoid using  0 as chunk #
        next_state = np.reshape(padded_state, [1, state_size])

       # next_state = np.reshape(correct_ranking, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    episode_rewards.append(total_reward)

#output learn result onto a table- state | best action learnedt
file_path = 'dqn_learned.csv'

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['State', 'Best Action'])  # Write header row

    for state, best_action in agent.state_to_best_action.items():
        writer.writerow([','.join(map(str, state)), best_action])



# Plotting
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Learning Progress')
plt.show()

