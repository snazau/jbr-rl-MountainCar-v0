import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set()


curr_dpi = 150


def plot_Q_table(Q_table, save_path=None):
	'''
	Q_table - 3dim numpy array that contains values of Q_table
	save_path - str, by default equal to None -> picture will be shown to user
	'''

	save_path_wo_ext, ext = os.path.splitext(save_path)
	position_state_amount, speed_state_amount, action_amount = Q_table.shape

	ax = sns.heatmap(np.argmax(Q_table, axis=2))
	if save_path is None:
		plt.show()
	else:
		plt.xlabel("speed")
		plt.ylabel("position")
		plt.savefig(save_path_wo_ext + "_actions" + ext, dpi=curr_dpi)
	plt.close()

	ax = sns.heatmap(np.amax(Q_table, axis=2))
	if save_path is None:
		plt.show()
	else:
		plt.xlabel("speed")
		plt.ylabel("position")
		plt.savefig(save_path_wo_ext + "_values" + ext, dpi=curr_dpi)
	plt.close()


def plot_avg_rewards(avg_rewards_indices, avg_rewards, save_path=None, xlabel="Episodes", ylabel="Avg reward"):
	'''
	plots y(x)
	avg_rewards_indices - x
	avg_rewards - y
	save_path - str, by default equal to None -> picture will be shown to user
	xlabel - str label for x axis
	ylabel - str label for y axis
	'''

	plt.plot(avg_rewards_indices, avg_rewards)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title('max=' + str(max(avg_rewards)))
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1, x2, -205, -90))
	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path, dpi=curr_dpi)
