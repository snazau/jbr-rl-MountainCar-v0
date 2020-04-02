import copy
import datetime
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import Memory
from dqn import DQN
import visualization


enviroment = gym.make('MountainCar-v0')
device = torch.device("cuda:0")
curr_date = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
preload = True


def initialization(hidden_size_1, hidden_size_2, activation):
	global device
	learning_model = DQN(hidden_size_1, hidden_size_2, activation)
	target_model = copy.deepcopy(learning_model)

	def init_weights(layer):
		if type(layer) == nn.Linear:
			nn.init.xavier_normal_(layer.weight)
	learning_model.apply(init_weights)

	learning_model = learning_model.to(device)
	target_model = target_model.to(device)

	return learning_model, target_model


def prepare_input(state):
	global device
	prepared_input = torch.tensor(state).to(device).float()  # (S, )
	prepared_input = prepared_input.unsqueeze(0)  # adding batch dim [1 x S]
	return prepared_input


def select_action(model, state, eps):
	if random.random() < eps:
		return random.randint(0, 2)
	prepared_input = prepare_input(state)
	with torch.no_grad():
		selected_action = model(prepared_input)  # [1 x A]
	selected_action = selected_action[0].max(0)
	selected_action = selected_action[1].view(1, 1).item()
	return selected_action


def train_batch(batch, learning_model, target_model, optimizer, discount_rate):
	global device
	learning_model.train()

	state, action, reward, state_new, done = batch
	state = torch.tensor(state).to(device).float()
	action = torch.tensor(action).to(device)
	reward = torch.tensor(reward).to(device).float()
	state_new = torch.tensor(state_new).to(device).float()
	done = torch.tensor(done).to(device)

	q_target = torch.zeros(reward.size()[0]).to(device).float()
	with torch.no_grad():
		q_target = target_model(state_new)
		q_target = q_target.max(1)[0].view(-1)
		q_target[done] = 0
	q_target = reward + discount_rate * q_target
	q_current = learning_model(state).gather(1, action.unsqueeze(1))

	# Huber loss
	loss = F.smooth_l1_loss(q_current, q_target.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()

	for param in learning_model.parameters():
		param.grad.data.clamp_(-1, 1)

	optimizer.step()
	return loss


def test_episode(model, render=False):
	global enviroment

	done = False
	accumulated_reward = 0

	state = enviroment.reset()
	episode_reward = 0
	while not done:
		if render:
			enviroment.render()
		action = select_action(model, state, 0)
		state, reward, done, info = enviroment.step(action)
		accumulated_reward += reward

	return accumulated_reward


def test(model, episode_amount):
	global enviroment
	model.eval()

	rewards = []
	for episode in range(episode_amount):
		reward = test_episode(model)
		rewards.append(reward)

	return np.mean(rewards)


def get_descrete_policy(model, sample_amount=50):
	global enviroment
	model.eval()

	position_min, speed_min = enviroment.observation_space.low
	position_max, speed_max = enviroment.observation_space.high
	action_amount = enviroment.action_space.n

	position_values = [position_min + ((position_max - position_min) / sample_amount) * x for x in range(sample_amount - 1)]
	position_values.append(position_max)
	speed_values = [speed_min + ((speed_max - speed_min) / sample_amount) * x for x in range(sample_amount - 1)]
	speed_values.append(speed_max)

	policy = np.zeros(shape=(sample_amount, sample_amount, action_amount))
	for i, position in enumerate(position_values):
		for j, speed in enumerate(speed_values):
		    prepared_input = prepare_input(np.array([position, speed]))
		    with torch.no_grad():
		    	curr_policy = model(prepared_input).cpu().numpy()  # [1 x A]
		    policy[i, j] = curr_policy

	return policy


def save_checkpoint(model, params, save_path):
	checkpoint = {
		"params": params,
		"state_dict": model.state_dict(),
	}
	torch.save(checkpoint, save_path)


def load_model(checkpoint_path):
	global device
	checkpoint = torch.load(checkpoint_path)
	params = checkpoint["params"]
	model = DQN(params["hs1"], params["hs2"], params["activation"])
	model.load_state_dict(checkpoint["state_dict"])
	model = model.to(device)
	print(checkpoint_path, "loaded")
	print_params(params)
	return model, params


def print_params(params):
	for key, value in zip(params.keys(), params.values()):
		print(key, "=", value)
	print()


def params_to_str(params):
	params_str = ""
	for key, value in zip(params.keys(), params.values()):
		params_str += key + "=" + str(value) + "_"
	params_str = params_str[:-1]
	return params_str


if __name__ == "__main__":
	checkpoints_dir = "./checkpoints"
	if not os.path.exists(checkpoints_dir):
		os.makedirs(checkpoints_dir)

	visualization_dir = "./DQL_visualization"
	if not os.path.exists(visualization_dir):
		os.makedirs(visualization_dir)

	if preload is False:
		params = {
			"lr": 1e-3,
			"dr": 0.99,
			"hs1": 64,
			"hs2": 64,
			"activation": "prelu",
			"max_eps": 0.5,
			"min_eps": 0.1,
			"batch_size": 256,
			"reward_type": "mod",
			"step_amount": 100001,
			"sync_models": 1000,
			"buf_size": 5000,
		}
		print("Train config:")
		print_params(params)
		params_str = params_to_str(params)

		learning_model, target_model = initialization(params["hs1"], params["hs2"], params["activation"])

		render_every_n = 5000
		avg_test_rewards = []
		avg_test_rewards_indices = np.arange((params["step_amount"] // params["sync_models"]) + 1)
		eps_reduction = (params["max_eps"] - params["min_eps"]) / params["step_amount"]

		best_avg_reward = -float("Inf")
		best_model = None
		best_step = -1

		memory = Memory(params["buf_size"])
		optimizer = torch.optim.Adam(learning_model.parameters(), lr=params["lr"])

		training_visualization_dir = os.path.join(visualization_dir, curr_date)
		if not os.path.exists(training_visualization_dir):
			os.makedirs(training_visualization_dir)
		state = enviroment.reset()
		for step in range(params["step_amount"]):
			eps = params["max_eps"] - step * eps_reduction
			action = select_action(learning_model, state, eps)
			state_new, reward, done, info = enviroment.step(action)
			if params["reward_type"] == "mod":
				modified_reward = reward + 10 * abs(state_new[1]) + (state_new[0] if state_new[0] > 0.25 else 0)
			else:
				modified_reward = reward

			memory.push((state, action, modified_reward, state_new, done))
			if step > params["batch_size"]:
				train_batch(memory.sample(params["batch_size"]), learning_model, target_model, optimizer, params["dr"])

			if done:
				state = enviroment.reset()
				done = False
			else:
				state = state_new

			if step % params["sync_models"] == 0:
				target_model = copy.deepcopy(learning_model)

				avg_test_reward = test(target_model, episode_amount=10)
				avg_test_rewards.append(avg_test_reward)
				print("avg_test_reward =", avg_test_reward, "on step =", step)

				if avg_test_reward > best_avg_reward:
					best_avg_reward = avg_test_reward
					best_model = copy.deepcopy(target_model)
					best_step = step

				done = False
				state = enviroment.reset()

				discrete_policy = get_descrete_policy(target_model)
				save_path = os.path.join(training_visualization_dir, str(step) + ".png")
				visualization.plot_Q_table(discrete_policy, save_path=save_path)

			if step % render_every_n == 0:
				episode_reward = test_episode(target_model, render=True)
				print("reward during rendered episode =", episode_reward, "on step =", step)
				done = False
				state = enviroment.reset()

		print("final avg_test_reward", test(target_model, episode_amount=100))
		print("best avg_test_reward", test(best_model, episode_amount=100), "on step =", best_step)

		save_path = os.path.join(checkpoints_dir, "DQN_" + curr_date + ".pth.tar")
		params["best_step"] = best_step
		save_checkpoint(best_model, params, save_path=save_path)
		model = best_model

		save_path = os.path.join(visualization_dir, "DQN_" + curr_date + "_rewards.png")
		visualization.plot_avg_rewards(avg_test_rewards_indices, avg_test_rewards, save_path=save_path, xlabel="Steps(thousands)", ylabel="Reward(avg)")

		discrete_policy = get_descrete_policy(model)
		save_path = os.path.join(visualization_dir, "DQN_" + curr_date + ".png")
		visualization.plot_Q_table(discrete_policy, save_path=save_path)
	else:
		# checkpoint_name = "DQN_lr=0.001_dr=0.99_hs1=64_hs2=128_max_eps=0.5_min_eps=0.1_batch_size=256_reward_type=mod_step_amount=100001_sync_models=1000_buf_size=5000.pth.tar"
		# checkpoint_name = "DQN_Apr02_15-17-34.pth.tar"  # step = 85k prelu avg ~ 104
		checkpoint_name = "DQN_Apr02_14-15-12.pth.tar"  # step = 69k prelu avg ~ 100
		pretrained_path = os.path.join(checkpoints_dir, checkpoint_name)
		model, params = load_model(pretrained_path)

		test_episode_amount = 100
		avg_test_reward = test(model, episode_amount=test_episode_amount)
		print("avg_test_reward =", avg_test_reward, "test_episode_amount =", test_episode_amount)

		discrete_policy = get_descrete_policy(model)
		visualization.plot_Q_table(discrete_policy, save_path="./" + checkpoint_name + ".png")

	print()
	print("Playing time c:")
	play_episode_amount = 10
	for episode in range(play_episode_amount):
		print("Episode {}/{}".format(episode + 1, play_episode_amount))
		reward = test_episode(model, render=True)
		print("Reward =", reward)
		print()
	enviroment.close()
	print("OK")
