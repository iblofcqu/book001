import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve as conv


def plot_state_action_values(env, value, ax=None):
    """
    可选参数，表示绘图将生成的坐标轴。如果不提供，将创建一个新的图形和坐标轴。

    Args:
        env: 环境对象
        value:  Q-table，表示为形状为 (n_states, n_actions) 的数组
        ax: 可选参数，表示绘图将生成的坐标轴。如果不提供，将创建一个新的图形和坐标轴。

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots()

    for a in range(env.n_actions):
        ax.plot(range(env.n_states), value[:, a],
                marker='o', linestyle='--')
    ax.set(xlabel='States', ylabel='Values')

    ax.legend(['R', 'U', 'L', 'D'], loc='lower right')


def plot_quiver_max_action(env, value, ax=None):
    """
    生成在每个状态下显示最大价值或最大概率动作

    Args:
        env: 环境对象
        value: Q-table，表示为形状为 (n_states, n_actions) 的数组。
        ax: 可选参数，表示绘图将生成的坐标轴。如果不提供，将创建一个新的图形和坐标轴。

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots()

    big_x = np.tile(np.arange(env.dim_x), [env.dim_y, 1]) + 0.5
    big_y = np.tile(np.arange(env.dim_y)[::-1][:, np.newaxis],
                    [1, env.dim_x]) + 0.5
    which_max = np.reshape(value.argmax(axis=1), (env.dim_y, env.dim_x))
    which_max = which_max[::-1, :]
    big_u = np.zeros(big_x.shape)
    big_v = np.zeros(big_x.shape)
    big_u[which_max == 0] = 1
    big_v[which_max == 1] = 1
    big_u[which_max == 2] = -1
    big_v[which_max == 3] = -1

    ax.quiver(big_x, big_y, big_u, big_v)
    ax.set(
        title='Maximum value/probability actions',
        xlim=[-0.5, env.dim_x + 0.5],
        ylim=[-0.5, env.dim_y + 0.5],
    )
    ax.set_xticks(np.linspace(0.5, env.dim_x - 0.5, num=env.dim_x))
    ax.set_xticklabels(["%d" % x for x in np.arange(env.dim_x)])
    ax.set_xticks(np.arange(env.dim_x + 1), minor=True)
    ax.set_yticks(np.linspace(0.5, env.dim_y - 0.5, num=env.dim_y))
    # code too long in a line
    y_tick_labels = np.arange(0, env.dim_y * env.dim_x, env.dim_x)
    ax.set_yticklabels(list(map(lambda x: str(int(x)), y_tick_labels)))
    ax.set_yticks(np.arange(env.dim_y + 1), minor=True)
    ax.grid(which='minor', linestyle='-')


def plot_heatmap_max_val(env, value, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if value.ndim == 1:
        value_max = np.reshape(value, (env.dim_y, env.dim_x))
    else:
        value_max = np.reshape(value.max(axis=1), (env.dim_y, env.dim_x))
    value_max = value_max[::-1, :]

    im = ax.imshow(value_max,
                   aspect='auto',
                   interpolation='none',
                   cmap='afmhot')
    ax.set(title='Maximum value per state')
    ax.set_xticks(np.linspace(0, env.dim_x - 1, num=env.dim_x))
    ax.set_xticklabels(["%d" % x for x in np.arange(env.dim_x)])
    ax.set_yticks(np.linspace(0, env.dim_y - 1, num=env.dim_y))
    if env.name != 'windy_cliff_grid':
        y_tick_labels = np.arange(0, env.dim_y * env.dim_x, env.dim_x)
        ticks_after_handle = list(map(lambda x: str(int(x)), y_tick_labels))
        ax.set_yticklabels(ticks_after_handle[::-1])
    return im


def plot_rewards(n_episodes, rewards, average_range=10, ax=None):
    """
    生成显示每个训练过程的累积的总奖励

    Args:
        n_episodes: 智能体训练次数
        rewards: 训练过程智能体获得的总奖励
        average_range: 用于平滑奖励曲线的参数
        ax: 可选参数，表示绘图将生成的坐标轴。如果不提供，将创建一个新的图形和坐标轴。

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots()

    smoothed_rewards = (conv(rewards, np.ones(average_range), mode='same')
                        / average_range)

    ax.plot(range(0, n_episodes, average_range),
            smoothed_rewards[0:n_episodes:average_range],
            marker='o',
            linestyle='--')
    ax.set(xlabel='Episodes', ylabel='Total reward')


def plot_performance(env, value, reward_sums, n_episodes: int):
    """
    调用定义的画图函数，生成强化学习训练过程和结果的可视化
    Args:
        env: 环境对象
        value: Q-table，表示为形状为 (n_states, n_actions) 的数组
        reward_sums: 训练过程智能体获得的总奖励
        n_episodes: 训练总次数

    Returns:

    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    plot_state_action_values(env, value, ax=axes[0, 0])
    plot_quiver_max_action(env, value, ax=axes[0, 1])
    plot_rewards(n_episodes, reward_sums, ax=axes[1, 0])
    im = plot_heatmap_max_val(env, value, ax=axes[1, 1])
    fig.colorbar(im)

    fig.savefig('results_figure.png', dpi=300)
