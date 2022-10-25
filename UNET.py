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
from torchmetrics.functional import dice_score

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # print(x.shape, "=x")
        # print(conv1.shape, "=conv1")
        # print(conv2.shape, "=conv2")
        # print(conv3.shape, "=conv3")
        upconv3 = self.upconv3(conv3)
        # print(upconv3.shape, "=upconv3")
        
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        
        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )

        return expand
    

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.to(torch.device("cuda"))

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.to(torch.device("cuda"))
                y = y.to(torch.device("cuda"))
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y).detach().cpu().numpy()

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                # if step % 10 == 0:
                #     # clear_output(wait=True)
                #     print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                #     # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            
        
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(torch.where(outputs > 0.5, 1.0, 0.0).permute((2,3,1,0)).squeeze().detach().cpu().numpy(), cmap='gray')        
        # ax[1].imshow(y.permute((1,2,0)).detach().cpu().numpy(), cmap='gray')
        # plt.show()
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    


def acc_metric(predb, yb):
    dice = DiceLoss()
    return dice(predb, yb)


def set_seed(seed=777):
    '''Set seed for every random generator that used in project'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    
    set_seed()
    
    unet = UNET(3,1)
    unet.to(torch.device("cuda"))
    
    df = pd.read_csv("D:\PROJECTS\Python_projects\PythonMain\data.csv")
    data_train, data_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=True)
    
    
    dataset_train = WaveDataset(data_train)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    
    dataset_test = WaveDataset(data_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1)
    
    
    loss_fn = DiceLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.1)
    
    train_loss, valid_loss = train(unet, dataloader_train, dataloader_test, loss_fn, opt, acc_metric, epochs=1000)
    
    # plt.plot(train_loss)
    # plt.plot(valid_loss)
    # plt.show()