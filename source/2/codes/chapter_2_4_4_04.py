from typing import Dict

import numpy as np


def q_learning(state: int, action: int, reward: float, next_state: int,
               value: np.ndarray, params: Dict):
    """
    Q-learning

    Args:
        state: 当前状态标识符
        action: 执行的动作
        reward: 接收到的奖励
        next_state: 转换到的状态标识符
        value: 当前价值函数，形状为(n_states, n_actions)
        params:默认参数字典

    Returns:
        更新后的价值函数，形状为(n_states, n_actions)

    """
    # 当前状态-动作对的q值
    q = value[state, action]
    # 找到下一个状态的最大Q值
    if next_state is None:
        max_next_q = 0
    else:
        max_next_q = np.max(value[next_state])
    # 计算时序差分 TD error
    td_error = reward + params['gamma'] * max_next_q - q
    # 更新Q值
    value[state, action] = q + params['alpha'] * td_error

    return value
