import matplotlib.pyplot as plt

from chapter_2_3_2_06 import Generator
from chapter_2_3_2_04 import generate_random

# 测试生成器
gen = Generator()
# 绘制6张结果图 —— 未经过训练的生成器
f, ax_arr = plt.subplots(2, 3, figsize=(16, 8))
for i in range(2):
    for j in range(3):
        outputs = gen(generate_random(1))
        img = outputs.detach().numpy().reshape(28, 28)
        ax_arr[i, j].imshow(img, interpolation='None', cmap='Blues')
plt.show()
