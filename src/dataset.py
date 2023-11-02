import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, u_feat, i_feat, target):
        self.u_feat = u_feat
        self.i_feat = i_feat
        self.target = target

    def __getitem__(self, index):
        return self.u_feat[index], self.i_feat[index], self.target[index]

    def __len__(self):
        return len(self.target)
