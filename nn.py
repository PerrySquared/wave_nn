import matplotlib.pyplot as plt
from torch import nn
from data import WaveDataset
from torch.utils.data import Dataset, DataLoader
from  sklearn.model_selection import train_test_split
import pandas as pd
import torch
import time
import os
import numpy as np
import random
import torch.nn.functional as F
import segmentation_models_pytorch.losses as smpl
import segmentation_models_pytorch as smp


# class UNET(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.conv1 = self.contract_block(in_channels, 32, 7, 3)
#         self.conv2 = self.contract_block(32, 64, 3, 1)
#         self.conv3 = self.contract_block(64, 128, 3, 1)

#         self.upconv3 = self.expand_block(128, 64, 3, 1)
#         self.upconv2 = self.expand_block(64*2, 32, 3, 1)
#         self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

#     def __call__(self, x):

#         # downsampling part
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         upconv3 = self.upconv3(conv3)
        
#         upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
#         upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

#         return upconv1

#     def contract_block(self, in_channels, out_channels, kernel_size, padding):

#         contract = nn.Sequential(
#             torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#                                  )

#         return contract

#     def expand_block(self, in_channels, out_channels, kernel_size, padding):
        
#         expand = nn.Sequential(
#             torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
#                             )

#         return expand 


def set_seed(seed=777):
    '''Set seed for every random generator that used in project'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.unsqueeze(dim=1).to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


if __name__ == '__main__':
    
    set_seed()


    data_train, data_test = train_test_split(pd.read_csv("data.csv"), train_size=0.9, test_size=0.1, shuffle=False)


    dataset_train = WaveDataset(data_train)
    dataset_test = WaveDataset(data_test)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    aux_params = dict(
        dropout = 0.15,
        classes=1,
        activation="tanh"
    )
    
    model = smp.Unet(encoder_name="resnet152", encoder_weights="imagenet", encoder_depth=5,
                     in_channels=3, decoder_attention_type="scse", aux_params=aux_params).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7000, eta_min=1e-6)
    DiceLoss = smpl.DiceLoss(mode='multilabel')

    def criterion(input_masks, target_masks):
        return DiceLoss(input_masks,target_masks)


    metric = np.zeros((0, 2))
    image_number = 0
    
    epochs = 40
    
    for i in range(epochs):
        train_loss = np.array([])
        valid_loss = np.array([])

        model.train()

        for source, mask in dataloader_train:
            optimizer.zero_grad()

            source, mask = source.to(device), mask.to(device)

            out = model(source)
            mask = mask.unsqueeze(dim=1).float()
            loss = criterion(out[0].float(), mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss = np.append(train_loss, loss.item())

        model.eval()

        with torch.no_grad():
            
            for source, mask in dataloader_test:
                    source, mask = source.to(device), mask.to(device)
                    out = model(source)
                    
                    # print(out[0], "\n=====", out[1])
                    temp_out = out[0].cpu()

                    valid_loss = np.append(valid_loss, dice_coef(mask, out[0].float()).cpu().detach().numpy())
                    
            image_number+=1
            image = temp_out.permute(2, 3, 1, 0).squeeze().squeeze()
            image = torch.where(image > 0.5, 1.0, 0.0)
            image = torch.nn.functional.softmax(image, 1)
            image = image > 0.9
            
            plt.imshow(image)
            plt.savefig(os.path.join("data_out", str(image_number)), dpi=300)
        
        plt.close()  
            
        metric = np.append(metric, [[np.mean(train_loss), np.mean(valid_loss)]], axis=0)

        print(
            f'\n---> {i+1}\033[95m LR:\033[0m {optimizer.param_groups[0]["lr"]:3e}' +
            f'\n|\033[94m Loss_train:\033[0m {metric[-1, 0]:.5}' +
            f'\n|\033[96m Acc_valid:\033[92m {metric[-1, 1]:.5}' +
            '\n----------------------\033[0m')

    # plt.plot(epochs, [val[1] for val in metric])
    # plt.show()
