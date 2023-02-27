import random

class MyDataset:
    def __init__(self,all_data,batch_size,shuffle=True):
        self.all_data = all_data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.cursor = 0      # 左边界

    # 魔术方法（特定条件下自动触发的函数）

    def __iter__(self):   # 只有第一次会触发，蓑衣需要返回一个具有__next__的对象
        print("======hello _iter_======")
        if self.shuffle:
            random.shuffle(self.all_data)
        self.cursor = 0    #游标重置
        return self

    def __next__(self):
        if self.cursor >= len(self.all_data):
            raise StopIteration
        batch_data = self.all_data[self.cursor : self.cursor+self.batch_size]
        self.cursor += self.batch_size
        return batch_data

if __name__ == "__main__":
    all_data = [1,2,3,4,5,6,7]
    batch_size = 2
    epoch = 3
    shuffle = True

    dataset = MyDataset(all_data,batch_size,shuffle)

    for e in range(epoch):
        for batch_data in dataset:  # 把一个对象放在for上时，会自动调用这个对象的__iter__()，但只会在第一个循环触发
            print(batch_data)