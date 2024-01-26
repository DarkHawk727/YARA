from dqn2_1 import RecurrentDQNAgent, DummyEnv
from BERTKeywordEmbedder import BERTKeywordEmbedder
import numpy as np
import matplotlib.pyplot as plt
import csv # state | best action learned

# Initialize BERTKeywordEmbedder
keyword_embedder = BERTKeywordEmbedder()

# Initialize environment
n = 10  # total action
k = 3   # top k choice
env = DummyEnv(n, k)

# Initialize agent
state_size = 300  # Assuming initial state size is 300
action_size = n   # Total number of possible actions
agent = RecurrentDQNAgent(state_size, action_size)

# Lists to store episode rewards
episode_rewards = []

# Lists to store episode rewards
episode_rewards = []

for episode in range(10):
    state = env.reset()
    padded_state = np.zeros((1, state_size, state_size))  # Initialize with zeros
    padded_state[0, :len(state), :len(state)] = state 
    state = padded_state
    total_reward = 0  # Track total reward for this episode

    for time in range(5):
        action = agent.act(state)
        correct_ranking, reward, done = env.step(action)
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
  
#output learn result onto a table- state | best action learned
file_path = 'dqn_learned_train.csv'

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['State', 'Best Action'])  # Write header row

    for state, best_action in agent.state_to_best_action.items():
        writer.writerow(['"' + state + '"', best_action])


# plot
# Plot learning progress after each episode
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Progress')
    plt.show()





