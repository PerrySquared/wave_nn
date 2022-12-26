from tqdm import tqdm
from data import WaveDataset
from torch.utils.data import Dataset, DataLoader
from  sklearn.model_selection import train_test_split
import torch
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
    
def recursive_search(comb, current_column, current_row):
    
    possibilities = 0
    row_len, column_len = comb.shape

    if current_column > 0:
        if comb[current_column - 1, current_row] == 2:
            return 1
        if comb[current_column - 1, current_row] == 1:
            possibilities += 1
            comb[current_column, current_row] = 0
            return recursive_search(comb, current_column - 1, current_row)
        
    if current_column < column_len - 1:
        if comb[current_column + 1, current_row] == 2:
            return 1 
        if comb[current_column + 1, current_row] == 1:
            possibilities += 1  
            comb[current_column, current_row] = 0
            return recursive_search(comb, current_column+1, current_row)     
        
        
    if current_row > 0:
        if comb[current_column, current_row - 1] == 2:
            return 1   
        if comb[current_column, current_row - 1] == 1:
            possibilities += 1  
            comb[current_column, current_row] = 0
            return recursive_search(comb, current_column, current_row-1)      
        
    if current_row < row_len - 1:
        if comb[current_column, current_row + 1] == 2:
            return 1
        if comb[current_column, current_row + 1] == 1:
            possibilities += 1  
            comb[current_column, current_row] = 0
            return recursive_search(comb, current_column, current_row+1)
    
    if possibilities == 0:
        return 0


def defection_check(source, predict):

    source, predict = source.cpu().squeeze(), predict.cpu()
    start_row, start_column = 0, 0

    predict = torch.where(predict > -5, 1.0, 0.0)

    row_len, column_len = source.shape
    
    for row in range(row_len):
        for column in range(column_len):
            if source[column, row] > 0.666 and source[column, row] < 0.668:
                start_row = row
                start_column = column
    
    comb = source + predict
    current_row, current_column = start_row, start_column
    
    defect = recursive_search(comb, current_column, current_row)
     
    return defect  

if __name__ == '__main__':
    
    set_seed()
    
    data_size = 10000
    batch_size = 64

    data_test = pd.read_csv("data.csv", nrows=data_size)

    dataset_test = WaveDataset(data_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

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

    model.to(device)
    model.eval()
    valid_accuracy = np.array([])
    path_lengts_avg_mask = np.array([])
    path_lengts_avg_pred = np.array([])
    
    with torch.no_grad():
        
        for source, mask in tqdm(dataloader_test):
            
            source, mask = source.to(device), mask.to(device)
            out = model(source)
            out = torch.squeeze(out[0])
       
            for i in range(len(mask)):
                
                name += 1
                
                torch.save(out[i].clone().detach(), os.path.join("./data_out_trained/tensors", str(name)))

                path_lengts_avg_mask = np.append(path_lengts_avg_mask, mask[i].cpu().sum())
                # path_lengts_avg_pred = np.append(path_lengts_avg_pred, torch.where(out[i] > -5, 1.0, 0.0).cpu().sum())
                
                # if(mask[i].sum() > 0):     
                #     valid_accuracy = np.append(valid_accuracy, defection_check(source[i], out[i]))
                
                # code below was used to generate images, but commented for the sake of benchmarking
                
                # source = torch.squeeze(source).cpu()              
                # mask = torch.squeeze(mask).cpu()
                
                # image = out.cpu()
                # image = torch.where(image > 0, 1.0, 0.0)

                # plt.imshow(mask[i] + source[i])
                # plt.savefig(os.path.join("data_out_trained/images", str(name) + "_mask"), dpi=300)
                # plt.imshow(image[i] + source[i])
                # plt.savefig(os.path.join("data_out_trained/images", str(name) + "_predicted"), dpi=300)
                # plt.close()  
                
print(valid_accuracy.mean())   
print(path_lengts_avg_mask.mean())           
# print(path_lengts_avg_pred.mean())    
