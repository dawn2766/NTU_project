# import torch
# import torch.nn as nn
# import math

# # 假设有两个样本，四个类别
# N = 2
# C = 4

# # 创建一个随机输入数据和对应的目标标签
# input = torch.randn(N, C)  # 输入数据形状为 (2, 4)
# input[1,:]=0
# target = torch.randint(0, C, (N,))  # 目标标签形状为 (2,)

# # print("input:",input)
# # print("target:",target)

# # 使用 nn.CrossEntropyLoss 计算损失
# loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(input, target)

# # print("输入参数形状示例：")
# # print("输入数据形状：", input.shape)
# # print("目标标签形状：", target.shape)
# # print("loss：", loss)

# class MyIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, start, end):
#         super(MyIterableDataset).__init__()
#         assert end > start, "this example code only works with end >= start"
#         self.start = start
#         self.end = end
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:  # single-process data loading, return the full iterator
#             iter_start = self.start
#             iter_end = self.end
#         else:  # in a worker process
#             # split workload
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             # print(worker_id)
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)
#         return iter(range(iter_start, iter_end))
# # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
# ds = MyIterableDataset(start=3, end=7)

# if __name__ == "__main__":
#     # Single-process loading
#     print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

#     # Mult-process loading with two worker processes
#     # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
#     print(list(torch.utils.data.DataLoader(ds, num_workers=2,batch_size=2)))

#     # With even more workers
#     print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
#     print("##########")
#     for i in torch.utils.data.DataLoader(ds, num_workers=12):
#         print(i)

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, label):
        print("init")
        self.data = data
        self.label=label

    def __len__(self):
        print(len(self.data))
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        print(sample.shape)
        return sample,self.label

# 创建自定义数据集对象
data = torch.randn(4, 2, 4)
custom_dataset = CustomDataset(data,'q')

# 创建数据加载器 DataLoader
batch_size = 2
shuffle = True

data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=shuffle)

# 遍历数据加载器来获取数据
for batch_data,i in data_loader:
    print("Batch data:", batch_data,i)

print(len(data_loader))
