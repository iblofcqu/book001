class CliffWorld:
    """
    40个状态 (4x10格子世界)
    状态到网格的映射如下:
    30 31 32 ... 39
    20 21 22 ... 29
    10 11 12 ... 19
    0 1 2 ... 9
    状态0为起点(S)，状态9为目标点(G)
    动作0、1、2、3分别对应右、上、左、下
    从状态9(目标点G)走出去即结束会话
    状态11-18执行动作3将掉入悬崖并返回状态0，同时获得-100的奖励
    在非目标点的任何状态将获得-1的奖励
    在边界处向边界外移动将保持原地
    """

    def __init__(self):
        # 世界名称
        self.name = "cliff_world"
        # 状态数
        self.n_states = 40
        # 动作数
        self.n_actions = 4
        # x 轴维度
        self.dim_x = 10
        # y 轴维度
        self.dim_y = 4
        # 初始状态
        self.init_state = 0

    def get_outcome(self, state, action):
        """
        定义智能体的动作和状态更新函数
        :param state:
        :param action:
        :return:
        """
        # 智能体进入状态9(目标点)，本轮结束，奖励为0
        if state == 9:
            reward = 0
            next_state = None
            return next_state, reward

        # 默认奖励为-1，使得智能体寻找最短路径以获得最大奖励
        reward = -1
        # 动作0位向右移动，状态+1
        if action == 0:
            next_state = state + 1
            # 当智能体到达右边界，状态保持不变
            if state % 10 == 9:
                next_state = state
            # 当智能体进入悬崖，本轮结束，奖励为-100
            elif state == 0:
                next_state = None
                reward = -100
        # 动作0为向上移动，状态+10
        elif action == 1:
            next_state = state + 10
            # 智能体到达上边界，状态保持不变
            if state >= 30:
                next_state = state
        elif action == 2:
            next_state = state - 1
            if state % 10 == 0:
                next_state = state
        elif action == 3:
            next_state = state - 10
            if 11 <= state <= 18:
                next_state = None
                reward = -100
            elif state <= 9:
                next_state = state
        else:
            print("Action must be between 0 and 3.")
            next_state = None
            reward = None
        return int(next_state) if next_state is not None else None, reward

    def get_all_outcomes(self):
        """
        定义环境输出的状态和奖励
        :return:
        """
        outcomes = {}
        # 遍历所有的状态动作对，得到特定状态下采取特定动作得到的状态和奖励。
        # 该方法将为每个状态-动作对添加一个条目，
        # 其中键是状态和动作的元组，
        # 值是包含(1,next_state,reward)元组的列表。
        # 所有条目都添加完后，该方法返回outcomes字典
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state, reward = self.get_outcome(state, action)
                outcomes[state, action] = [(1, next_state, reward)]
        return outcomes
