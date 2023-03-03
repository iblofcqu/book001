import torch

x = torch.tensor([2])
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
y = torch.mul(w, x)
z = torch.add(y, b)

# x，w，b叶子节点的值
print("x,w,b的require_grad值为：{},{},{}".format(x.requires_grad,
                                                w.requires_grad,
                                                b.requires_grad))

# 查看叶子节点、非叶子节点的其他属性：
print("y，z的requires_grad值分别为：{},{}".format(y.requires_grad,
                                                z.requires_grad))
# 非叶子节点的requires_grad值
# 说明：因与w，b有依赖关系，故y，z的requires_grad属性也是：True,True
print("x,w,b,y,z的叶子节点属性：{},{},{},{},{}".format(x.is_leaf,
                                                      w.is_leaf,
                                                      b.is_leaf,
                                                      y.is_leaf,
                                                      z.is_leaf))
# 查看各节点是否叶子节点
print("x,w,b的grad_fn属性：{},{},{}".format(x.grad_fn,
                                           w.grad_fn,
                                           b.grad_fn))
# 叶子节点的grad_fn属性
# 说明：因x，w，b为用户创建的，故grad_fn属性为None
print("y,z的叶子节点属性：{},{}".format(y.grad_fn == None,
                                       z.grad_fn == None))
# y，z是否为叶子节点
# 自动求导，实现梯度的反向传播：
z.backward()
# 基于z张量进行梯度反向传播，如果需要多次使用backward
# 需要修改参数retain_graph为True，此时梯度是累加的
print(z)
print("w,b,x的梯度分别为:{},{},{}".format(w.grad,
                                          b.grad, x.grad))
# 说明：x是叶子节点但它无须求导，故其梯度为None
print("非叶子节点y,z的梯度分别为:{},{}".format(y.retain_grad(),
                                               z.retain_grad()))
# 说明：当执行backward之后，非叶子节点的梯度会自动清空
