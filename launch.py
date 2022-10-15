import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from data_prepare import testloader
from models import GCNN, AttGNN

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = GCNN()
model.load_state_dict(torch.load("/Users/jiadonglou/Desktop/Benjamin/PPI_GNN/Human_features/GCN.pth")) #path to load the model
model.to(device)
model.eval()
predictions = torch.Tensor()
labels = torch.Tensor()

