from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import streamlit as st
import seaborn as sns


def show_path(paths, reward, episode_reward):
	r = np.copy(reward)

	fig = plt.figure(figsize=(1.5, 1.5))
	ax = fig.add_subplot(111)
	plt.axis('off')

	grid_holder = st.empty()
	grid = ax.imshow(r, cmap='tab20c', interpolation='nearest')

	index = 0
	for episode in paths[::10]:
		episode = np.insert(episode, -1,  [-1,  -1], axis=0)
		for i, j in episode:
			r = np.copy(reward)
			r[i, j] = -4

			grid.set_data(r)
			plt.title(f"Episode:{index}  Reward:{episode_reward[index]}")
			plt.draw()
			grid_holder.pyplot(fig)
			time.sleep(0.01)

		time.sleep(1)
		index += 10


def main():

	reward15 = np.array([[-1, -10, -1,  -1,  -10,  -1, -1, -1, -1, -1, -1, -10, -1, -1, -1],
						 [-1, -10, -10, -1,  -10,  -1, -10, -10, -10, -10, -10, -10, -1, -10, -1], 
						 [-1, -1,  -1,  -1,  -10, -1, -10, -1, -10, -1, -1, -10, -1, -10, -1], 
						 [-1, -10, -10, -10, -10,  -1, -10, -1, -10, -1, -10, -10, -1, -10, -1], 
						 [-1, -1,  -1,  -1,  -1,  -1, -10, -1, -10, -1, -1, -10, -1, -10, -1], 
						 [-1, -10, -10, -10,  -1,  -10, -10, -1, -10, -1, -10, -10, -1, -10, -1], 
						 [-1, -10, -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -10, -1, -10, -1], 
						 [-1, -10, -10, -10,  -1,  -10, -10, -10, -10, -10, -1, -10, -1, -10, -1], 
						 [-1, -1,  -1,  -1,  -1,  -10, -1, -1, -1, -1, -1, -1, -1, -10, -1], 
						 [-1, -10, -1,  -10,  -1,  -10, -10, -10, -10, -10, -10, -10, -1, -1, -1], 
						 [-1, -10, -1,  -10,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -10, -1], 
						 [-10,-10, -1,  -10,  -1,  -10, -10, -10, -10, -10, -10, -10, -1, -10, -1], 
						 [-1, -1,  -1,  -10,  -1,  -1, -10, -1, -10, -10, -1, -10, -1, -10, -1], 
						 [-1, -10, -10, -10,  -1,  -10, -10, -1, -10, -10, -1, -10, -10, -10, -1], 
						 [-1, -1,  -1,  -1,  -1,  -10, 10, -1, -1, -1, -1, -1, -1, -1, -1]])

	reward5 = np.array([[-1, -10, -1, -1, -1],
					   [-1, -1, -1, -10, -1],
					   [-1, -10, -10, -10, -1],
					   [-1, -10, 10, -1, -1],
					   [-1, -1, -1, -1, -1]])

	reward4 = np.array([[-1, -10, -1, -1],
					   [-1, -1, -1, -1],
					   [-10, -10, -10, -1],
					   [-10, 10, -1, -1]])

	reward3 = np.array([[-1, -10, -1],
					   [-1, -1, -1],
					   [-1, -10, 1 -1]])

	st.sidebar.write("# Gridworld")

	reward = st.sidebar.selectbox("Choose Grid", ["15x15", "5x5", "4x4", "3x3"])
	episodes = st.sidebar.slider("Number of Episodes", 1, 1000, 100, 10)
	maxsteps = st.sidebar.slider("Max Steps per Episode", 10, 100, 10, 10)
	lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.70, 0.1)
	decay = st.sidebar.slider("Decay Rate", 0.01, 1.0, 0.01, 0.01)
	random_start = st.sidebar.checkbox("Random Start")

	if reward == "15x15":
		reward = reward15
		size = 15
	elif reward == "5x5":
		reward = reward5
		size = 5
	elif reward == "4x4":
		reward = reward4
		size = 4
	else:
		reward = reward3
		size = 3
	
	if st.sidebar.button('Train Agent'):
		agent = Agent(reward, episodes=episodes, maxsteps=maxsteps, lr=lr, decay=decay, random_start=random_start)
		agent.train()
		show_path(agent.paths, reward, agent.episode_rewards)

		# fig = plt.figure(figsize=(1,1))
		# ax = fig.add_subplot(111)
		# plt.axis('off')
		# grid_holder = st.empty()
		# table = np.amax(agent.qtable, axis=1)
		# table = np.reshape(table, (15,15))
		# table[table == 0] = -100
		# grid = ax.imshow(table)
		# grid_holder.pyplot(fig)
		# print(table)

		table = np.amax(agent.qtable, axis=1)
		table = np.reshape(table, (size,size))
		table[table == 0] = -100


		policy = np.argmax(agent.qtable, axis=1).astype(str)
		policy = np.reshape(policy, (size, size))
		policy[policy == '0'] = 'L'
		policy[policy == '1'] = 'U'
		policy[policy == '2'] = 'R'
		policy[policy == '3'] = 'D'
		# print(policy)
		for i in range(size):
			for j in range(size):
				if table[i, j] == -100:
					if reward[i, j] != -10:
						table[i, j] = 10
						policy[i, j] = 'W'
					else:
						policy[i, j] = ''

		grid_holder = st.empty()
		fig, ax = plt.subplots()
		ax= sns.heatmap(table, annot=policy, fmt="")
		grid_holder.pyplot(fig)


		grid_holder = st.empty()
		fig, ax = plt.subplots()
		ax = sns.lineplot(list(range(len(agent.episode_rewards))), agent.episode_rewards)
		ax.set(xlabel='Episode', ylabel='Reward')
		grid_holder.pyplot(fig)


if __name__ == "__main__":
	main()
