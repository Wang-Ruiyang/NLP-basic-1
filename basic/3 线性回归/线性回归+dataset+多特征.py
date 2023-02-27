import numpy as np

class DataSet:
    def __init__(self,years,floors,prices,lr,batch_size,shuffle=True):
        self.years = years
        self.floors = floors
        self.prices = prices
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(self.years)


class DataLoader:
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0

        self.index = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        ind = self.index[self.cursor : self.cursor+self.dataset.batch_size]
        x = self.dataset.years[ind]
        y = self.dataset.floors[ind]
        z = self.dataset.prices[ind]
        self.cursor += self.dataset.batch_size

        return x, y, z

if __name__ == "__main__":

    a = 1
    b = -1
    c = 0

    epoch = 10000
    batch_size = 2
    lr = 0.1
    shuffle = True

    years = np.array([i for i in range(2000, 2022)])
    years = (years - 2000) / 22

    floors = np.array([i for i in range(23, 1, -1)])
    floors = floors / 23

    prices = np.array(
        [10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000, 23000, 26000, 35000,
         30000, 40000, 45000, 52000, 50000, 60000])
    prices = prices / 60000

    dataset = DataSet(years,floors,prices,lr,batch_size,shuffle)

    for e in range(epoch):
        for year,floor,price in dataset:

            predict = a * year + b * floor + c
            loss = np.sum((predict - price) ** 2)

            delta_a = 2 * (predict - price) * year
            delta_b = 2 * (predict - price) * floor
            delta_c = 2 * (predict - price)

            a = a - np.sum(delta_a) * lr
            b = b - np.sum(delta_b) * lr
            c = c - np.sum(delta_c) * lr

    print(f"k={a},b={b},c={c}")

    while True:
        year = (float(input("请输入年份：")) - 2000)/22
        floor = float(input("请输入楼层：")) / 23
        print("预测房价：", (a * year +b * floor + c) * 60000)