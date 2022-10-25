import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class WaveDataset(Dataset):
    def __init__(self, files_path: pd.DataFrame, show_progress: bool=True, train: bool = True, resave: bool = False):
        super().__init__()
        self.source = []
        self.target = []

        pbar = tqdm(total = len(files_path), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') if show_progress else None

        for _, row in files_path.iterrows():
            self.source.append(torch.load(row.iloc[1]))
            self.target.append(torch.load(row.iloc[2]))
            
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()
            
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx: int):
        return self.source[idx], self.target[idx]



# torch.set_printoptions(edgeitems=1000)
# source, target = next(iter(dataloader))

# test1 = source.squeeze().numpy()
# f = plt.figure()
# f.add_subplot(1,2,1)
# plt.imshow(test1, cmap='gray') 
# test2 = target.squeeze().numpy()
# f.add_subplot(1,2,2)
# plt.imshow(test2, cmap='gray')
# plt.show()