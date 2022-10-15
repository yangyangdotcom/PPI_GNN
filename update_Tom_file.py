import numpy as np
import pandas as pd
import os
# tmp = np.load("npy_ar_update.csv",allow_pickle=True)

path = '/Users/jiadonglou/Desktop/Benjamin/PPI_GNN/Human_features/processed' 
files = [os.path.splitext(filename)[0] for filename in os.listdir(path)]

df = pd.read_csv("npy_ar_update.csv")

df = df.loc[df["2"].isin(files)]
df = df.loc[df["5"].isin(files)]
df.drop(columns = df.columns[0], axis = 1, inplace= True)

#read name .pt file from processed into an array

# print(files)
# print(df)

npy = df.to_numpy()
print(npy)
np.save("update_numpy",npy,allow_pickle=True)
# df.to_csv("update_numpy.csv")

