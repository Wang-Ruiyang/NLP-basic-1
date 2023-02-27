import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))    # 利用np对numpy的每个数都单独进行这样的操作


if __name__ == "__main__":

    # [毛发长，腿长]
    dogs = np.array([[8.9,12],[9,11],[10,13],[9.9,11.2],[12.2,10.1],[9.8,13],[8.8,11.2]],dtype = np.float32)   # 0
    cats = np.array([[3,4],[5,6],[3.5,5.5],[4.5,5.1],[3.4,4.1],[4.1,5.2],[4.4,4.4]],dtype = np.float32)        # 1
    labels = np.array([0]*7+[1]*7, dtype=np.int32).reshape(-1,1)
    # print(label)

    X = np.vstack((dogs,cats))    # np.vstack返回竖直堆叠后的数组
    # print(X)

    # np.random.normal能指定生成的数据的均值和方差
    k = np.random.normal(0,1,size=(2,1))    # 这里规定均值为0，方差为1，2行1列
    b = 0    # 偏置值
    epoch = 1000
    lr = 0.05

    for e in range(epoch):
        p = X @ k + b    # 计算结果

        pre = sigmoid(p)      # pre的每个值都在0~1
        loss = -np.mean(labels * np.log(pre) + (1-labels) * np.log((1-pre)))     # loss是一个标量
        G = pre - labels

        delta_k = X.T @ G
        delta_b = np.sum(G)

        k = k - lr * delta_k
        b = b - lr * delta_b

        print(loss)

    while True:
        f1 = float(input("请输入毛发长："))
        f2 = float(input("请输入腿长："))

        text_x = np.array([f1,f2]).reshape(1,2)
        p = sigmoid(text_x @ k + b)
        if p>0.5:
            print("类别：猫")
        else:
            print("类别：狗")
