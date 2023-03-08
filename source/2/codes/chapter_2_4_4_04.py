from typing import Dict

import numpy as np


def q_learning(state: int, action: int, reward: float, next_state: int,
               value: np.ndarray, params: Dict):
    """
    Q-learning

    Args:
        state: 当前
        action:
        reward:
        next_state:
        value:
        params:

    Returns:

    """
    q = value[state, action]

    if next_state is None:
        max_next_q = 0
    else:
        max_next_q = np.max(value[next_state])

    td_error = reward + params['gamma'] * max_next_q - q
    value[state, action] = q + params['alpha'] * td_error

    return value
