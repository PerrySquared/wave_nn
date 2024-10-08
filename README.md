# Neural Network for Lee Algorithm


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Performance](#performance)
* [Setup and usage](#setup-and-usage)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)



## General Information
- This project is created to train a neural network based on a generated dataset that takes two combined rain matrices (to roughly emulate how a proper chip layout would look like) and uses Lee Algorithm to make one route (if it exists) for each one of them
- Goal of this project is to see if a trained neural network can improve the speed of Lee Algorithm with minimal amount of defects
- Currently an archived repository


## Technologies Used
- Anaconda
- Python 3.10
- PyTorch 1.13.0
- Pandas
- NumPy
- Scikit-learn


## Features
- The ability to generate datasets with different parameters
- Trained model can be saved and exported
- Performance of the algorithm and trained model can be measured 
- Support for creating images on different stages of the process to help with visualisation of data


## Screenshots
<p align="center">
  <img width="600" height="600" src="./other/Figure_3.png">
</p>
<p align="center"> Source matrix example </p>
<p align="center">
  <img width="600" height="500" src="./other/prediction.png">
</p>
<p align="center"> Prediction example </p>


## Performance
<p> 13 times runtime imrovement at the cost of having breaks in some paths (~10% of tests), </p>
<p> must be used alongside algotithms with 100% accuracy to compensate for unsuccesful preditions </p>
<p> future improvements might be possible, but to make this method viable on its own a new approach is probably needed </p>

## Setup and usage
Requirements are listed under  [Technologies Used](#technologies-used)

To start using the project you will need to:
1. Download the repository
2. Install anaconda terminal
4. Generate the dataset via autogen.py
5. Train the neural network via nn.py
6. Best model checkpoint data will be saved in a .pth file which can be used for testing in saved_model_test.py

