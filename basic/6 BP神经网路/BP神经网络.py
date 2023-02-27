import matplotlib.pyplot as plt
import numpy as np
import struct

def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)

def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)

# 将label变成举证（60000*10）
def make_one_hot(labels,class_num=10):
    result = np.zeros((len(labels),class_num))
    for index,lab in enumerate(labels):     # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
        result[index][lab] = 1
    return result

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    ex = np.exp(x)    # 整个矩阵的每个元素都求指数
    sum_ex = np.sum(ex, axis=1, keepdims=True)     # 按行求指数结果的和，axis=1表示按行，keepdims=True表示保持原来形状
    return ex/sum_ex


if __name__ == "__main__":
    train_datas = load_images("data/train-images.idx3-ubyte") / 255     # (60000, 784)
    train_label = make_one_hot(load_labels("data/train-labels.idx1-ubyte"),10)          # #  (60000,)
    # print(train_datas.shape,train_label.shape)
    # print(train_label[0])    # 输出一行

    # 测试集不做one_hot
    test_datas = load_images("data/t10k-images.idx3-ubyte") / 255
    test_label = load_labels("data/t10k-labels.idx1-ubyte")

    epoch = 100
    batch_size = 600     # 一次性处理多少图片
    lr = 0.01

    hidden_num = 256     # 隐层大小
    w1 = np.random.normal(0,1,size=(784,hidden_num))
    w2 = np.random.normal(0,1,size=(hidden_num,10))

    batch_times = int(np.ceil(len(train_datas) / batch_size))     # np.ceil 向上取整

    for e in range(epoch):
        for batch_index in range(batch_times):

            batch_x = train_datas[batch_index * batch_size : (batch_index + 1) * batch_size]     # 按行为单位取出，每次取batch_size行
            batch_label = train_label[batch_index * batch_size : (batch_index + 1) * batch_size]

            # forward
            h = batch_x @ w1
            sig_h = sigmoid(h)
            p = sig_h @ w2
            pre = softmax(p)

            # 计算数据
            loss = -np.mean(batch_label * np.log(pre))/batch_size     # 求平均loss

            # backward
            G2 = (pre - batch_label)/batch_size     # backward都和G2有关，G2会因为batch_size过大而梯度爆炸，这里除以batch_size可以避免梯度爆炸，
            delta_w2 = sig_h.T @ G2
            delta_sig_h = G2 @ w2.T
            delta_h = delta_sig_h * sig_h * (1 - sig_h)    # 1-sig_h是sig_h中每个元素都被1减
            delta_w1 = batch_x.T @ delta_h

            # 更新梯度
            w1 = w1 - lr * delta_w1
            w2 = w2 - lr * delta_w2

        # 利用测试集计算精确度
        h = test_datas @ w1
        sig_h = sigmoid(h)
        p = sig_h @ w2
        pre = softmax(p)     #pre是一个10000行1列的向量
        # print(pre.shape)
        pre = np.argmax(pre, axis=1)     # 取一行最大值的下标，最终的pre是一个10000行1列的向量
        # print(pre.shape)

        acc = np.sum(pre==test_label)/10000

        print(acc)

    # 画图
    # t = train_datas[1107]
    # plt.imshow(t.reshape(28,28))
    # plt.show()
    # print(train_label[1037])