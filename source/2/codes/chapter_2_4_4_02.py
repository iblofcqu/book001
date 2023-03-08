import numpy as np


def epsilon_greedy(q: np.ndarray, epsilon: float) -> int:
    """
    Epsilon贪心策略: 以概率(1-epsilon)选择最大值动作，以epsilon概率随机选择
    :param q: 动作值的数组
    :param epsilon:随机选择动作的概率
    :return:选择的动作
    """
    # 以概率(1-epsilon)选择最大值动作
    if np.random.random() > epsilon:
        action = np.argmax(q)
    else:
        # 以 epsilon 概率随机选择动作
        action = np.random.choice(len(q))

    return action
