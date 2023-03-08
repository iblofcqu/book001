from typing import Tuple

import numpy as np

from chapter_2_4_4_02 import epsilon_greedy


def learn_environment(env, learning_rule, params, max_steps: int,
                      n_episodes: int) -> Tuple[np.ndarray, int]:
    """
    以概率(1-epsilon)选择最大值动作，以epsilon概率随机选择
    :param env: 环境对象，特指CliffWorld
    :param learning_rule:一个基于观察更新价值函数的函数
    :param params:学习规则和探索策略中使用的参数字典
    :param max_steps:代表智能体在一个训练过程中可以采取的最大步数
    :param n_episodes:用于训练的代数
    :return:更新后的Q价值函数,shape 为(n_states, n_actions) 和 训练过程的总奖励数
    """
    # 初始化Q-table，创建一维数组（env.n_states, env.n_actions）且元素值均为1
    value = np.ones((env.n_states, env.n_actions))
    # 开始智能体学习过程
    reward_sums = np.zeros(n_episodes)
    # 开始训练循环
    for episode in range(n_episodes):
        # 初始化状态
        state = env.init_state
        reward_sum = 0
        for t in range(max_steps):
            # 根据epsilon贪心策略选择下一个动作
            action = epsilon_greedy(value[state], params['epsilon'])
            # 观察采取的动作得到的环境反馈
            next_state, reward = env.get_outcome(state, action)
            # 更新Q-table数值
            value = learning_rule(state, action, reward, next_state,
                                  value, params)
            # 计算总奖励
            reward_sum += reward
            # 定义训练终止条件
            if next_state is None:
                break
            state = next_state
        # 记录每一次训练过程的总奖励
        reward_sums[episode] = reward_sum

    return value, reward_sums
