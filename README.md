A solution for [mountain car problem](https://gym.openai.com/envs/MountainCar-v0/) via Q-learning with algorithmic and deep approaches.

## Contents
* [Requirements](#requirements)
* [How to run](#how-to-run)
* [DQL results](#dql-results)
* [QL results](#ql-results)

## Requirements
* **PyTorch** ([instructions](https://pytorch.org/get-started/locally/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **gym** ([instructions](https://github.com/openai/gym#installation))
* **seaborn** ([instructions](https://seaborn.pydata.org/installing.html))

## How to run
* Clone this repo to your local computer
* Install all required dependencies
* ???
* Type in console: 
* * `python deep_Q_learning.py` if you want to run deep-approach version
* * `python Q_learning.py` if you want to run algorithmic-approach version
* You're good now

## DQL results
Results with deep approach with following parameters:

* step_amount = 100001
* learning_rate = 0.001
* discount_rate = 0.99
* max_eps = 0.5
* min_eps = 0.1
* batch_size = 256
* sync_models = 1000
* best_step = 69000
* activation = prelu

*Avg reward:*
![](static/DQN_Apr02_14-15-12_rewards.png)
*Policy, 0 - move left, 1 - do nothing, 2 - move right:*
![](static/DQN_Apr02_14-15-12_actions.png)
*Corresponding values:*
![](static/DQN_Apr02_14-15-12_values.png)

## QL results
Results with algorithmic approach with following parameters:

* episode_amount = 40001
* learning_rate = 0.2
* discount_rate = 0.9
* max_eps = 0.5
* min_eps = 0.1
* grid_size = 80 * 60

*Avg reward:*
![](static/QL_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1.png)
*Policy, 0 - move left, 1 - do nothing, 2 - move right:*
![](static/Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1.png)
*Corresponding values:*
![](static/Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1_values.png)
