import numpy as np
import matplotlib

from chapter_2_4_4_01 import CliffWorld
from chapter_2_4_4_04 import q_learning
from chapter_2_4_4_05 import plot_performance
from chapter_2_4_4_03 import learn_environment

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(1)

params = {
    'epsilon': 0.1,
    'alpha': 0.1,
    'gamma': 1.0,
}

n_episodes = 500
max_steps = 1000

env = CliffWorld()

results = learn_environment(env, q_learning, params, max_steps, n_episodes)
value_qlearning, reward_sums_qlearning = results

plot_performance(env, value_qlearning, reward_sums_qlearning, n_episodes)
