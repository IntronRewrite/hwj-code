import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optimizer

batch_size = 16
lr = 1e-4
max_epochs = 100

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
