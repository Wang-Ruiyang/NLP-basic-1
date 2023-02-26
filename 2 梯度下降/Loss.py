from tqdm import trange     # 显示进度条

epoch = 10000
lr = 0.05
x = 3    # 初始值
label = 0

for e in trange(epoch):
    pre = (x-2) ** 2
    loss = (pre-label) ** 2
    delta_x = 2 * (pre-label)*(x-2)
    x = x - delta_x * lr

print(x)

