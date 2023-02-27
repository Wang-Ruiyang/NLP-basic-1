import numpy as np


class MyDataSet:
    def __init__(self, all_data, batch_size, shuffle=True):
        self.all_data = all_data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):  # 只有第一次会触发
        print("======hello _iter_======")
        return DataLoader(self)

    def __len__(self):
        return len(self.all_data)


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0  # 左边界

        self.index = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle == True:
            np.random.shuffle(self.index)

    def __next__(self):
        if self.cursor >= len(self.dataset.all_data):
            raise StopIteration
        ind = self.index[self.cursor: self.cursor + self.dataset.batch_size]  # ind是个列表
        batch_data = self.dataset.all_data[ind]  # batch_data也是list
        self.cursor += self.dataset.batch_size
        return batch_data


if __name__ == "__main__":
    all_data = np.array([1, 2, 3, 4, 5, 6, 7])
    batch_size = 2
    epoch = 3
    shuffle = True

    dataset = MyDataSet(all_data, batch_size, shuffle)

    for e in range(epoch):
        for batch_data in dataset:
            print(batch_data)