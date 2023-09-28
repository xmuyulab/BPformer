import numpy as np
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, x_mRNA, y_label):
        self.x_mRNA = x_mRNA
        self.y_label = y_label
        self.x_mRNA = self.x_mRNA.astype(np.float32)

    def __len__(self):
        return self.x_mRNA.shape[0]

    def __getitem__(self, item):
        x_mRNA = self.x_mRNA[item, :]
        y_label = self.y_label[item]
        return x_mRNA, y_label