import numpy as np

class DataSet:
    def __init__(self,years,prices,k,b,lr,batch_size,shuffle=True):
        self.years = years
        self.prices = prices
        self.k = k
        self.b = b
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(years)


class DataLoader:
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0
        self.index = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle==True:
            np.random.shuffle(self.index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        ind = self.index[self.cursor : self.cursor+self.dataset.batch_size]
        x = self.dataset.years[ind]
        y = self.dataset.prices[ind]
        self.cursor += self.dataset.batch_size

        return x, y

if __name__ == "__main__":

    epoch = 10000
    k = 1
    b = 1
    lr = 0.1
    batch_size = 2
    shuffle = True

    years = np.array([i for i in range(2000, 2022)])
    years = (years - 2000) / 22
    prices = np.array(
        [10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000, 23000, 26000, 35000,
         30000, 40000, 45000, 52000, 50000, 60000]) / 60000

    dataset = DataSet(years,prices,k,b,lr,batch_size,shuffle)

    for e in range(epoch):
        for year,price in dataset:

            predict = k * year + b
            loss = (predict - price) ** 2

            delta_k = 2 * (k * year + b - price) * year
            delta_b = 2 * (k * year + b - price)

            k = k - np.sum(delta_k) * lr
            b = b - np.sum(delta_b) * lr

    print(f"k={k},b={b}")

    while True:
        year = (float(input("请输入年份：")) - 2000)/22
        print("预测房价：", (k * year +b) * 60000)