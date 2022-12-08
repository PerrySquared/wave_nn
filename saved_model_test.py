from tqdm import tqdm
from torch import nn
from data import WaveDataset
from torch.utils.data import Dataset, DataLoader
from  sklearn.model_selection import train_test_split
import torch
import sys
import os
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smpl



def set_seed(seed=777):
    # Set seed for every random generator that used in project
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    
    set_seed()
    
    data_size = 1000
    batch_size = 64

    data_test = pd.read_csv("data.csv", nrows=data_size)

    dataset_test = WaveDataset(data_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print(device)
    
    aux_params = dict(
        dropout = 0.70,
        classes=1,
        activation="tanh"
    )
    
    model = smp.Unet(encoder_name="vgg16", encoder_weights="imagenet", encoder_depth=5,
                     in_channels=1, decoder_attention_type="scse", aux_params=aux_params).to(device)

    
    
    checkpoint = torch.load('best_model.pth')
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    name = 0
    image_number = 0

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        
        for source, mask in tqdm(dataloader_test):
            
            source, mask = source.to(device), mask.to(device)
            
            out = model(source)
            out = torch.squeeze(out[0])
       
            for i in range(len(mask)):
                
                name += 1

                torch.save(out[i].clone().detach(), os.path.join("./data_out_trained/tensors", str(name)))
                
                
                # code below was used to generate images, but commented for the sake of benchmarking
                
                
                # mask = torch.squeeze(mask).cpu()
                # image = out.cpu()
                # image = torch.where(image > 0.5, 1.0, 0.0)

                # plt.imshow(mask[i])
                # plt.savefig(os.path.join("data_out_trained/images", str(name) + "_mask"), dpi=300)
                # plt.imshow(image[i])
                # plt.savefig(os.path.join("data_out_trained/images", str(name) + "_predicted"), dpi=300)
                # plt.close()  
        
            

