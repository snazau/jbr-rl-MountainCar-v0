import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import visualization


enviroment = gym.make('MountainCar-v0')
preload = True


def get_index_on_grid(grid_values, value):
	index = np.argmin(np.abs(np.array(grid_values) - value))
	return index


def train_episode(Q_table, learning_rate, discount_rate, eps, reward_type="normal", speed_coef=10, render=False):
	global enviroment

	position_min, speed_min = enviroment.observation_space.low
	position_max, speed_max = enviroment.observation_space.high
	grid_position_amount, grid_speed_amount, action_amount = Q_table.shape

	position_values = [position_min + ((position_max - position_min) / grid_position_amount) * x for x in range(grid_position_amount)]
	position_values -= position_min
	speed_values = [speed_min + ((speed_max - speed_min) / grid_speed_amount) * x for x in range(grid_speed_amount)]
	speed_values -= speed_min

	done = False
	accumulated_reward = 0
	position, speed = enviroment.reset()

	while not done:
		if render is True:
			enviroment.render()

		position_index = get_index_on_grid(position_values, position - position_min)
		speed_index = get_index_on_grid(speed_values, speed - speed_min)

		# Epsilon greedy strategy
		# with eps probability will choose random action
		# with (1 - eps) probability will choose action via best current policy
		if np.random.random() <= eps:
			action = np.random.randint(0, action_amount)
		else:
			action = np.argmax(Q_table[position_index, speed_index])

		state_new, reward, done, info = enviroment.step(action)
		accumulated_reward += reward

		position_new = state_new[0]
		speed_new = state_new[1]
		position_new_index = get_index_on_grid(position_values, position_new - position_min)
		speed_new_index = get_index_on_grid(speed_values, speed_new - speed_min)

		# Updating Q_table
		# Bellman Equation
		# Q'(s, a) = (1 - lr) * Q(s, a) + lr * (r + d * Q(s', argmax a' : Q(s', a')))
		if reward_type == "modified":
			modified_reward = reward + speed_coef * abs(speed_new) + (position_new if position_new > 0.25 else 0)
		else:
			modified_reward = reward

		if done and position_new >= 0.5:
			Q_table[position_index, speed_index, action] = modified_reward
		else:
			Q_table[position_index, speed_index, action] += learning_rate * (modified_reward + discount_rate * np.max(Q_table[position_new_index, speed_new_index]) - Q_table[position_index, speed_index, action])

		position = position_new
		speed = speed_new

	return accumulated_reward


def train(episode_amount, learning_rate, discount_rate, max_eps, min_eps, reward_type, speed_coef):
	global enviroment

	position_min, speed_min = enviroment.observation_space.low
	position_max, speed_max = enviroment.observation_space.high

	# Building Q-table
	# it's necessary to descretize continuous values
	grid_position_amount = 80
	grid_speed_amount = 60
	action_amount = enviroment.action_space.n

	Q_table = np.random.uniform(low=-1, high=1, size=(grid_position_amount, grid_speed_amount, action_amount))

	eps_reduction = (max_eps - min_eps) / episode_amount
	eps = max_eps
	rewards = []
	avg_rewards = []
	avg_rewards_indices = []
	for episode in range(episode_amount):
		render = False

		if (episode + 1) % 500 == 0:
			render = True

		reward = train_episode(Q_table, learning_rate, discount_rate, eps, reward_type=reward_type, speed_coef=speed_coef, render=render)
		rewards.append(reward)

		# Epsilon decay
		eps = max(min_eps, max_eps - episode * eps_reduction)

		# Print & collect info
		if (episode + 1) % 500 == 0:
			print("Episode =", episode + 1, "reward =", reward, "avg_reward =", np.mean(rewards[-100:]))

		if (episode + 1) % 50 == 0 and episode > 100:
			avg_rewards.append(np.mean(rewards[-100:]))
			avg_rewards_indices.append(episode + 1)

	return Q_table, avg_rewards_indices, avg_rewards


def test_episode(Q_table, render=False):
	global enviroment

	position_min, speed_min = enviroment.observation_space.low
	position_max, speed_max = enviroment.observation_space.high
	grid_position_amount, grid_speed_amount, action_amount = Q_table.shape

	position_values = [position_min + ((position_max - position_min) / grid_position_amount) * x for x in range(grid_position_amount)]
	position_values -= position_min
	speed_values = [speed_min + ((speed_max - speed_min) / grid_speed_amount) * x for x in range(grid_speed_amount)]
	speed_values -= speed_min

	done = False
	accumulated_reward = 0
	position, speed = enviroment.reset()

	while not done:
		if render is True:
			enviroment.render()
		position_index = get_index_on_grid(position_values, position - position_min)
		speed_index = get_index_on_grid(speed_values, speed - speed_min)

		action = np.argmax(Q_table[position_index, speed_index])

		state_new, reward, done, info = enviroment.step(action)
		accumulated_reward += reward

		position = state_new[0]
		speed = state_new[1]

	return accumulated_reward


def test(Q_table, episode_amount):
	global enviroment

	rewards = []
	for episode in range(episode_amount):
		reward = test_episode(Q_table)
		rewards.append(reward)

	return np.mean(rewards)


def best_parameters_search(parameters):
	'''
	enviroment - gym env
	parameters - dictionary with content like:
	parameters = {
		"episode_amount": [20001],
		"learning_rate": [0.1],
		"discount_rate": [0.9, 0.99],
		"reward_type": ["normal"],
		"speed_coef": [10],
		"max_eps": [0.5],
		"min_eps": [0.0, 0.05, 0.1],
	}
	'''

	global enviroment

	solution_reward = -110
	solution_parameters = {
		"reward": [],
		"episode_amount": [],
		"learning_rate": [],
		"discount_rate": [],
		"reward_type": [],
		"speed_coef": [],
		"max_eps": [],
		"min_eps": [],
	}

	best_Q_table = None
	best_params = {}
	best_avg_test_reward = -float("Inf")
	for episode_amount in parameters["episode_amount"]:
		for learning_rate in parameters["learning_rate"]:
			for discount_rate in parameters["discount_rate"]:
				for reward_type in parameters["reward_type"]:
					for speed_coef in parameters["speed_coef"]:
						for max_eps in parameters["max_eps"]:
							for min_eps in parameters["min_eps"]:
								if max_eps >= min_eps:
									print("episode_amount =", episode_amount)
									print("learning_rate =", learning_rate)
									print("discount_rate =", discount_rate)
									print("reward_type =", reward_type)
									print("speed_coef =", speed_coef)
									print("max_eps =", max_eps)
									print("min_eps =", min_eps)

									Q_table, avg_rewards_indices, avg_rewards = train(episode_amount, learning_rate, discount_rate, max_eps, min_eps, reward_type, speed_coef)
									params_str = 'episodes=' + str(episode_amount) + '_lr=' + str(learning_rate) + '_dr=' + str(discount_rate) + '_rt=' + reward_type + '_sc' + str(speed_coef) + '_max_eps=' + str(max_eps) + '_min_eps=' + str(min_eps)

									visualization.plot_avg_rewards(avg_rewards_indices, avg_rewards, save_path=os.path.join(visualization_dir, "QL_" + params_str + ".png"))
									visualization.plot_Q_table(Q_table, save_path=os.path.join(visualization_dir, "Q_table_" + params_str + ".png"))

									avg_test_reward = test(Q_table, episode_amount=100)
									if avg_test_reward > best_avg_test_reward:
										best_avg_test_reward = avg_test_reward
										best_Q_table = copy.deepcopy(Q_table)
										best_params = {
											"episode_amount": episode_amount,
											"learning_rate": learning_rate,
											"discount_rate": discount_rate,
											"reward_type": reward_type,
											"speed_coef": speed_coef,
											"max_eps": max_eps,
											"min_eps": min_eps,
										}
									print("avg_test_reward =", avg_test_reward)

									if avg_test_reward >= solution_reward:
										solution_parameters["reward"].append(avg_test_reward)
										solution_parameters["episode_amount"].append(episode_amount)
										solution_parameters["learning_rate"].append(learning_rate)
										solution_parameters["discount_rate"].append(discount_rate)
										solution_parameters["reward_type"].append(reward_type)
										solution_parameters["speed_coef"].append(speed_coef)
										solution_parameters["max_eps"].append(max_eps)
										solution_parameters["min_eps"].append(min_eps)

									print()

	return solution_parameters, best_params, best_avg_test_reward, best_Q_table


def print_parameters_search_result(solution_parameters, best_params, best_avg_test_reward):
	print("Solution sets:")
	for reward, episode_amount, learning_rate, discount_rate, reward_type, speed_coef, max_eps, min_eps in zip(solution_parameters["reward"], solution_parameters["episode_amount"], solution_parameters["learning_rate"], solution_parameters["discount_rate"], solution_parameters["reward_type"], solution_parameters["speed_coef"], solution_parameters["max_eps"], solution_parameters["min_eps"]):
		print("episode_amount", episode_amount)
		print("learning_rate", learning_rate)
		print("discount_rate", discount_rate)
		print("reward_type", reward_type)
		print("max_eps", max_eps)
		print("min_eps", min_eps)
		print()
	print()

	print("best_avg_test_reward =", best_avg_test_reward)
	print("Best params:")
	for key, value in zip(best_params.keys(), best_params.values()):
		print(key, "=", value)


def save_Q_table(Q_table, params, save_path):
	data = {
		"Q_table": Q_table,
		"params": params,
	}
	with open(save_path, "wb") as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_Q_table(path):
	with open(path, "rb") as f:
		data = pickle.load(f)
	print(path, "loaded")
	Q_table = data["Q_table"]
	params = data["params"]
	for key, value in zip(params.keys(), params.values()):
		print(key, "=", value)

	return Q_table, params


if __name__ == "__main__":
	visualization_dir = "./QL_visualization"
	if not os.path.exists(visualization_dir):
		os.makedirs(visualization_dir)

	Q_tables_dir = "./Q_tables"
	if not os.path.exists(Q_tables_dir):
		os.makedirs(Q_tables_dir)

	Q_table = None
	if preload is False:
		parameters = {
			"episode_amount": [40001],
			"learning_rate": [0.1, 0.2],
			"discount_rate": [0.9],
			"reward_type": ["modified"],
			"speed_coef": [10],
			"max_eps": [0.5, 0.8],
			"min_eps": [0.05, 0.1],
		}

		parameters = {
			"episode_amount": [301],
			"learning_rate": [0.1],
			"discount_rate": [0.9],
			"reward_type": ["modified"],
			"speed_coef": [10],
			"max_eps": [0.5],
			"min_eps": [0.05],
		}

		solution_parameters, best_params, best_avg_test_reward, best_Q_table = best_parameters_search(parameters)
		print_parameters_search_result(solution_parameters, best_params, best_avg_test_reward)
		save_Q_table(best_Q_table, best_params, os.path.join(Q_tables_dir, "Q_table.pickle"))
		Q_table = best_Q_table
	else:
		path = "Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1_values.pickle"
		Q_table, params = load_Q_table(os.path.join(Q_tables_dir, path))

	test_episode_amount = 100
	avg_test_reward = test(Q_table, test_episode_amount)
	print("avg_test_reward =", avg_test_reward, "test_episode_amount =", test_episode_amount)
	print()

	print("Playing time c:")
	play_episode_amount = 3
	for episode in range(play_episode_amount):
		print("Episode {}/{}".format(episode + 1, play_episode_amount))
		reward = test_episode(Q_table, render=True)
		print("Reward =", reward)
		print()

	enviroment.close()

	print("OK")
