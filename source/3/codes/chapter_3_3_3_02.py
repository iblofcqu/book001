class EarlyStopping:
    """
    如果在给定的耐心值之后验证损失没有改善，则停止训练
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        # 把每次的验证损失依次赋给score（取负值）
        # 这里需要注意，损失越小越好，这里取负，则越大越好，比较时如果大就更新best_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            # 当新score比best_score小，则继续训练，直至patience次数停止训练
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果在patience次数内的某次score大，则更新best_score，重新计数
            self.best_score = score
            self.counter = 0
