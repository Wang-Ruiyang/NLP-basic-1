import numpy as np
import pandas as pd


def get_data(file):
    # skiprows=1表示删除表头，names表示重新加上表头
    data = pd.read_csv(file,names=["y","x1","x2","x3","x4","x5","x6"],skiprows=1)
    # print(data)

    y = data["y"].values.reshape(-1,1)     # reshape(行数,列表)，-1表示不知道，1表示确定为1列。所以reshape(-1,1)表示变成1列数据
    mean_y = np.mean(y)      # 求和取平均值
    std_y = np.std(y)        # 求标准差，即方差开根。列表求方差，需要将每个数与平均值相减再平方，这些平方相加的结果就是方差。
    y = (y - mean_y) / std_y
    # print(y)

    x = data[[f"x{i}" for i in range(1,7)]].values     # values是直接将数据转化为numpy矩阵形式
    mean_x = np.mean(x,axis=0,keepdims = True)    # axis=0为纵向,axis=1为横向；keepidms=True表示保持其二维或者三维的特性(结果保持其原来维数)
    std_x = np.std(x,axis=0,keepdims=True)
    x = (x - mean_x) / std_x
    # print(x)

    return x,y,mean_y,std_y,mean_x,std_x


if __name__ == "__main__":
    X,y,mean_y,std_y,mean_x,std_x = get_data("上海二手房价.csv")

    K = np.random.random((6,1))     # 生成6行1列的浮点数，浮点数都是从0-1中随机,维度为2

    epoch = 10000
    lr = 0.001
    b = 1      # 偏置项

    for e in range(epoch):
        pre = X @ K      # 270*6 @ 6*1 = 270*1
        # print(pre)
        loss = np.sum((pre-y)**2)/len(X)    # y是270*1
        print(loss)

        G = (pre - y)/len(X)
        delta_k = X.T @ G
        delta_b = np.mean(G)         # 偏置项求导就是G的求平均

        K = K - lr * delta_k
        b = b - lr * delta_b

    while True:
        bedroom = (int(input("请输入卧室数量:")))
        ting = (int(input("请输入客厅数量:")))
        wei = (int(input("请输入卫生间数量:")))
        area = (int(input("请输入面积:")))
        floor = (int(input("请输入楼层:")))
        year = (int(input("请输入建成年份:")))

        test_x = (np.array([bedroom, ting, wei, area, floor, year]).reshape(1, -1) - mean_x) / std_x

        p = test_x @ K + b
        print("房价为: ", p * std_y + mean_y)