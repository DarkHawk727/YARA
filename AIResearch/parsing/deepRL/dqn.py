"""

Description: 
A DQN RL network simulating subsystem 4 from GPT+ deep RL application.


Design parameters: 
Reward and action: 
Agent action: produce rank to predict data chunk response(concatenated by file name if there are many), ranked by responses of Low input GPT. 
Enviornment feedback and reward: 

assign grades to each response from a, re-rank based on grade e= [a2,....] (n of them)

Action space:
a= [a1,a2...,an]
ordered, a1....an= whole number representing data chunk splitted to fed to low GPT. 

Bellman: (optimizing problem, best Q(s,a)=0)
Q(s,a)= R(s,a) +(gamma)max a' Q (s,a')  



Loss: 
Q(s,a)-  (R(s,a) +(gamma)max a' Q (s,a') )  
 
R(s,a)= (-1)* Kendall's Tau distance bwtween vector e and a  (Kendall's Tau distance: larger number means more discrepency) 
Q (s,a')= estimate of updated netword after 1 policy gradient iteration


Other DQN parameters:
Experiment replay buffer = 0: No replay buffer because each state assumed independent identically distributed (IID).



"""


import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Define the custom environment with API-based rewards
class CustomEnvironment(gym.Env):
    def __init__(self, api_url): # where specified with “api_url” will be replaced by GPT and token authentication.
        super(CustomEnvironment, self).__init__()
        self.api_url = api_url
        self.state = None
        self.observation_space = gym.spaces.MultiDiscrete([n1, n2, n3, ...])# Assuming a represents a list of integers, each specifying the number of possible states for each dimension of your state space: each means response chunk, ranked by predicted correctness. 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(state_dimension,))

    def reset(self):
        # Implement reset logic (initialize environment state)
        self.state = initial_state
        return self.state

    def step(self, action):
        # Implement the step function (perform action, call API, update state, and calculate reward)
        next_state = self.take_action(action)
        reward = self.get_reward(next_state)
        done = self.is_terminal_state(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def take_action(self, action):
        # Implement action execution logic here
        # Update the state based on the action
        # Example: self.state = update_state_with_action(self.state, action)
        pass

    def get_reward(self, state):
        # Call the API to get the reward based on the current state’s question
        reward = call_api_to_get_reward(self.api_url, state.question)
        return reward

    def is_terminal_state(self, state):
        # Implement terminal state detection logic
        # Return True if the state is terminal, otherwise False
        Pass

"""  no replay, because no batch buffer but keep as reference:
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
"""


# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque()  # Replay buffer of size 0 : state.problem assume independent before feeding to DQN
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential() # Sequential != forward passing, it means stacking neural layers such as input, hidden and output layer. 
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


# Initialize the custom environment and DQN agent
api_url = "your_api_endpoint_here"
state_dimension = 4  # Define the dimension of the state
initial_state = np.zeros(state_dimension)  # Define the initial state

env = CustomEnvironment(api_url)
state_size = state_dimension
action_size = 2  # Binary action space
agent = DQNAgent(state_size, action_size)



# Precompute all possible rankings
all_rankings = list(itertools.permutations(items))

n_episodes = 1000 # In our case, if no follow up question enabled (which is enable for now), each episode has 1 state transition. 1000 of them = m*1000 Q-A sections generated by High GPT + n*1000 chunk responses are generated by low GPT. It is highly recommended a careful prompt to ask GPT to respond efficiently. 

# DQN agent initialization
agent = DQNAgent(state_size, len(all_rankings))

# Training loop # Training the DQN agent # GPT puts batch_size = 32 because of experience replay. Mini batch is selected randomly from experience replay to break correlation while ensuring gradient update is efficient. It is not uncommon to leave minibatch = 0 if correlation is ruled out from the beginning. 

for episode in range(num_episodes): 
    state = env.reset()
    done = False
    while not done:
        # Select an action from the precomputed rankings
        action_index = agent.select_action(state)  # Choose an index from 0 to len(all_rankings)
        selected_ranking = all_rankings[action_index]

        # Perform the action in the environment and receive feedback
        next_state, reward, done, _ = env.step(selected_ranking)

        # Update the DQN agent's Q-network based on the feedback
        agent.train(state, action_index, reward, next_state)

        state = next_state