import os.path
import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from turtle import shape
from algorithm import Wave_Route
from random import uniform, randint


xpix, ypix = 64, 64


def validate_coords(pic, y, x):
    if pic[y][x] == 0:
        return True
    return False

def rain_gen():
    
    pic = np.zeros((64, 64)) # size of pic matrix
    max_drop_height = 28 # right border of drop height, min is always is 1
    drop_margin = 2 # margin around of drop, min is always is 1
    section_width = 64 # width of a section
    section_height = max_drop_height + drop_margin*2 + 1 # section height


    for section in range(math.ceil(pic.shape[-1]/section_height)):
        
        section_frenq = uniform(1, 2)
        pic[section * section_height, :] = np.array([[0]*section_width])
        
        for x in range(0, pic.shape[0], drop_margin):
            
            y_left = section_frenq * np.cos((x)*section_frenq)
            y_right = section_frenq * np.cos((x+1)*section_frenq)

            if y_left >=0 and y_right <= 0:
                start_drop_y = randint(section * section_height, (section + 1) * section_height - max_drop_height + drop_margin*2 + 1) + drop_margin
                end_drop_y = start_drop_y + randint(1, max_drop_height + 1)

                if end_drop_y > pic.shape[-1] - 1:
                    end_drop_y = pic.shape[-1] - 1
                
                pic[start_drop_y:end_drop_y, x] = np.array([[-1]*(end_drop_y - start_drop_y)])

    return pic

def generator():
    
    pic = (rain_gen() + rain_gen().transpose()).clip(-1, 0) 

    # block that picks and verifies the position of start and end of the route
    validation = False
    while validation == False:
        start_x = randint(0, xpix - 1)
        start_y = randint(0, ypix - 1)
        validation = validate_coords(pic, start_y, start_x)

    validation = False
    while validation == False:
        end_x = randint(0, xpix - 1)
        end_y = randint(0, ypix - 1)
        validation = validate_coords(pic, end_y, end_x)


    pic_copy = pic.copy()
    pic_copy[start_y][start_x] = -2
    pic_copy[end_y][end_x] = -3

    wave = Wave_Route(pic, start_y, start_x, end_y, end_x)
    pic_copy = np.abs(pic_copy)
    pic_copy = pic_copy/np.max(pic_copy) # normalization 
    
    return pic_copy, wave.output()[0], wave.output()[1] # task, solution, wave_number

 
path_task = "./data_task"
path_solution = "./data_solved"

files = []
wave_lengths = []

for i in tqdm(range(10000)):
    
    task, solution, wave_number = 0, 0, 0
    task, solution, wave_number = generator()

    wave_lengths.append(wave_number)
    name = i
    task = np.expand_dims(task, axis=-1)
    
    
    torch.save(torch.tensor(task, dtype = torch.float).permute((2,0,1)), os.path.join(path_task, str(name)))
    torch.save(torch.tensor(solution, dtype = torch.long), os.path.join(path_solution, str(name)))
    
    files.append([os.path.join(path_task, str(name)), os.path.join(path_solution, str(name))])
    

wave_lengths = np.array(wave_lengths)

df = pd.DataFrame(data=files, columns=['source', 'target'])
df.to_csv("data.csv")
