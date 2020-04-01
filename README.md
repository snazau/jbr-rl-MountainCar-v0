A solution for [mountain car problem](https://gym.openai.com/envs/MountainCar-v0/) via Q-learning with algorithmic and deep approaches.

## Contents
* [Requirements](#requirements)
* [Q-table results](#q-table-results)
* [DQN results](#dqn-results)

## Requirements
* **PyTorch** ([instructions](https://pytorch.org/get-started/locally/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **gym** ([instructions](https://github.com/openai/gym#installation))
* **seaborn** ([instructions](https://seaborn.pydata.org/installing.html))

## Q-table results
Results with algorithmic approach with following parameters:

* episode_amount = 40001
* learning_rate = 0.2
* discount_rate = 0.9
* max_eps = 0.5
* min_eps = 0.1
* grid_size = 80 * 60

avg_test_reward = -112.15 test_episode_amount = 100

![](static/QL_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1.png)
![](static/Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1.png)
![](static/Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1_values.png)

## DQN results
