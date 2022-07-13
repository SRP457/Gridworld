import numpy as np
import random


class Agent():
	def __init__(self, reward, episodes=1, maxsteps=10, lr=0.7, discount=0.99, explore=1, decay=0.01, minrate=0.01, random_start=False):
		size = reward.shape[0]
		self.reward = reward
		self.qtable = np.zeros((size**2, 4))
		self.episodes = episodes
		self.maxsteps = maxsteps
		self.lr = lr
		self.discount = discount
		self.exploration_rate = explore
		self.decay_rate = decay
		self.min_exploration_rate = minrate
		self.random_start = random_start


	def get_index(self, current_state):
		ind = 0
		size = self.reward.shape[0]
		for i in range(size):
			for j in range(size):
				if ([i, j] == current_state).all():
					return ind
				ind += 1


	def validate(self, action, current_state):
		if action == 0:	# left
			new_state = np.add(np.array(current_state), np.array([0, -1]))
		elif action == 1: # up
			new_state = np.add(np.array(current_state), np.array([-1, 0]))
		elif action == 2:	# right
			new_state = np.add(np.array(current_state), np.array([0, 1]))
		elif action == 3: # down
			new_state = np.add(np.array(current_state), np.array([1, 0]))

		i, j = new_state
		size = self.reward.shape[0]

		if i in range(size) and j in range(size):
			if self.reward[i, j] == -10:
				return [-1]
			return new_state
		return [-1]


	def train(self):
		alpha = self.lr
		gamma = self.discount
		exploration_rate = self.exploration_rate
		minrate = self.min_exploration_rate
		decay = self.decay_rate
		path = list()
		episode_rewards = list()

		for episode in range(self.episodes):
			path.append([])
			rewards = 0

			if self.random_start:
				size = self.reward.shape[0]
				current_state = np.random.randint(low=0, high=size, size=2)

				while self.reward[current_state[0], current_state[1]] != -1:
					current_state = np.random.randint(low=0, high=size, size=2)
			else:
				current_state = np.array([0, 0])

			for step in range(self.maxsteps):
				index = self.get_index(current_state)
				threshold = random.uniform(0, 1)
				new_state = [-1]
				valid = True
				
				while new_state[0] == -1:
					if threshold > exploration_rate:
						actions = self.qtable[index, :]
						if valid == False:
							actions[action] = -100
						action = np.argmax(actions)
					else:
						action = random.choice([0,1,2,3])

					new_state = self.validate(action, current_state)
					if new_state[0] == -1:
						valid = False

				q = self.qtable[index, action]
				i, j = new_state
				rt = self.reward[i, j]
				new_index = self.get_index(new_state)
				max_q = np.max(self.qtable[new_index, :])
				path[episode].append(list(new_state))

				self.qtable[index, action] = (1-alpha)*q + alpha*(rt + gamma*(max_q))
				current_state = new_state
				rewards += rt

				if rt >= 10:				
					break

			episode_rewards.append(rewards)
			exploration_rate = minrate + (1 - minrate) * np.exp(-decay*episode)
		
		self.paths = np.array(path)
		self.episode_rewards = episode_rewards
