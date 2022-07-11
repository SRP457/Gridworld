from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import streamlit as st


def show_path(paths, reward, episode_reward):
	r = np.copy(reward)

	fig = plt.figure(figsize=(3, 2))
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

	st.sidebar.write("# Gridworld\n")

	reward = st.sidebar.selectbox("Choose Grid", ["15x15", "5x5", "4x4", "3x3"])
	episodes = st.sidebar.slider("Number of Episodes", 1, 1000, 100, 10)
	maxsteps = st.sidebar.slider("Max Steps per Episode", 10, 100, 10, 10)
	lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.70, 0.1)
	decay = st.sidebar.slider("Decay Rate", 0.01, 1.0, 0.01, 0.01)

	if reward == "15x15":
		reward = reward15
	elif reward == "5x5":
		reward = reward5
	elif reward == "4x4":
		reward = reward4
	else:
		reward = reward3
	
	if st.sidebar.button('Train Agent'):
		agent = Agent(reward, episodes=episodes, maxsteps=maxsteps, lr=lr, decay=decay)
		agent.train()
		show_path(agent.paths, reward, agent.episode_rewards)


if __name__ == "__main__":
	main()
