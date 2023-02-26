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

# 将label变成矩阵（60000*10）
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

def get_datas():
    train_datas = load_images("data/train-images.idx3-ubyte") / 255     # (60000, 784)
    train_label = make_one_hot(load_labels("data/train-labels.idx1-ubyte"),10)          # #  (60000,)

    test_datas = load_images("data/t10k-images.idx3-ubyte") / 255
    test_label = load_labels("data/t10k-labels.idx1-ubyte")

    return train_datas,train_label,test_datas,test_label

class Linear:
    def __init__(self,in_num,out_num):
        self.weight = np.random.normal(0,1,size=(in_num, out_num))

    def forward(self,x):
        self.x = x
        return self.x @ self.weight

    def backward(self,G):
        delta_weight = self.x.T @ G
        delta_x = G @ self.weight.T
        self.weight -= lr * delta_weight     # 优化器的内容，梯度下降优化器 SGD
        return delta_x

    def __call__(self, x):    # 类似于重载
        return self.forward(x)

class Sigmoid:
    def forward(self,x):
        self.r = sigmoid(x)
        return self.r

    def backward(self,G):
        return G * self.r * (1-self.r)

    def __call__(self, x):
        return self.forward(x)

class Softmax:
    def forward(self,x):
        self.r = softmax(x)
        return self.r

    def backward(self,G):   # 传的是label
        return (self.r - G)/self.r.shape[0]    # batch_size就是第0个维度的shape

    def __call__(self, x):
        return self.forward(x)

class MyModel:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,x,label=None):
        for layer in self.layers:
            x = layer(x)
        self.x = x
        if label is not None:
            self.label = label
            loss = -np.mean(label * np.log(x)) / x.shape[0]
            return loss

    def backward(self):
        G = self.label
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    def __call__(self, *args):    # *args变参，数量不变
        return self.forward(*args)



if __name__ == "__main__":
    train_datas,train_label,test_datas,test_label = get_datas()

    # 定义参数
    epoch = 100
    batch_size = 600     # 一次性处理多少图片
    lr = 0.01
    hidden_num = 256     # 隐层大小

    model = MyModel([
        Linear(784,hidden_num),
        Sigmoid(),
        Linear(hidden_num,10),
        Softmax()
    ])

    batch_times = int(np.ceil(len(train_datas) / batch_size))     # np.ceil 向上取整

    for e in range(epoch):
        for batch_index in range(batch_times):

            x = train_datas[batch_index * batch_size : (batch_index + 1) * batch_size]     # 按行为单位取出，每次取batch_size行
            batch_label = train_label[batch_index * batch_size : (batch_index + 1) * batch_size]

            # forward
            loss = model.forward(x,batch_label)
            if batch_index%100==0:
                print(f"loss={loss:.3f}")
            model.backward()

            # backward && 更新梯度
            model.backward()

        # 利用测试集计算精确度
        x = test_datas
        model.forward(x)

        pre = np.argmax(model.x, axis=1)     # 取一行最大值的下标，最终的pre是一个10000行1列的向量
        acc = np.sum(pre==test_label)/10000

        print(f"acc={acc:.3f}")