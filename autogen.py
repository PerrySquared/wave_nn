import os.path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
from turtle import shape
from algorithm import Wave_Route
from perlin_noise import PerlinNoise


xpix, ypix = 64, 64


def validate_coords(pic, y, x):
    if pic[y][x] == 0:
        return True
    return False


def generator():
    
    noise = PerlinNoise(octaves=12)
    
    pic = np.zeros((0 ,xpix))
    for i in range(xpix):
        row = np.array([])
        for j in range(ypix):
        
            noise_val = noise([i/xpix, j/ypix])
            
            if noise_val < -0.1: # point of switch from obstacle to free space
                noise_val = -1
            else:
                noise_val = 0
            
            row = np.append(row, noise_val)
        pic = np.append(pic, [row], axis=0)

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

    return pic_copy, wave.output()

 
path_task = "D:\PROJECTS\Python_projects\PythonMain\data_task"
path_solution = "D:\PROJECTS\Python_projects\PythonMain\data_solved"

files = []

for i in tqdm(range(1000)):
    
    task, solution = 0, 0
    task, solution = generator()
    
    name = i
    # print(name, "\n")
    task = np.expand_dims(task, axis=-1)
    tmp = task
    task = np.append(task, tmp, axis = -1)
    task = np.append(task, tmp, axis = -1)
            
    torch.save(torch.tensor(task, dtype = torch.float).permute((2,0,1)), os.path.join(path_task, str(name)))
    torch.save(torch.tensor(solution, dtype = torch.long), os.path.join(path_solution, str(name)))
    
    files.append([os.path.join(path_task, str(name)), os.path.join(path_solution, str(name))])
    
    # task = pd.DataFrame(data=task.astype(float))
    # task.to_csv(os.path.join(path_task, str(name)+".csv"), sep=' ', header=False, float_format='%.2f', index=False, mode="+w")
    
    # solution = pd.DataFrame(data=solution.astype(float))
    # solution.to_csv(os.path.join(path_task, str(name)+".csv"), sep=' ', header=False, float_format='%.2f', index=False, mode="+w")

df = pd.DataFrame(data=files, columns=['source', 'target'])
df.to_csv("D:\PROJECTS\Python_projects\PythonMain\data.csv")

# # test1 = torch.load(os.path.join(path_task, str(0))).numpy()
# f = plt.figure()
# # f.add_subplot(1,2,1)
# # plt.imshow(test1, cmap='gray') 
# test2 = torch.load(os.path.join(path_solution, str(0))).numpy()
# f.add_subplot(1,2,2)
# plt.imshow(test2, cmap='gray')
# plt.show()