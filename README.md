A solution for [mountain car problem](https://gym.openai.com/envs/MountainCar-v0/) via Q-learning with algorithmic and deep approaches.

## Contents
* [Requirements](#requirements)
* [How to run](#how-to-run)
* [DQN results](#dqn-results)
* [Q-table results](#q-table-results)

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

## DQN results

## Q-table results
Results with algorithmic approach with following parameters:

* episode_amount = 40001
* learning_rate = 0.2
* discount_rate = 0.9
* max_eps = 0.5
* min_eps = 0.1
* grid_size = 80 * 60

avg_test_reward = -112.15 test_episode_amount = 100

*Avg reward*
![](static/QL_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1.png)
*Policy, 0 - move left, 1 - do nothing, 2 - move right*
![](static/Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1.png)
*Q-table values*
![](static/Q_table_episodes=40001_lr=0.2_dr=0.9_rt=modified_sc10_max_eps=0.5_min_eps=0.1_values.png)
