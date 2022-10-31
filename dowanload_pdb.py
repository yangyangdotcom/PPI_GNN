import os
import requests

path = '/Users/jiadonglou/Desktop/Benjamin/PPI_GNN/Human_features/raw' 
files = [os.path.splitext(filename)[0] for filename in os.listdir(path)]
# print (files)

for item in range(len(files)):
    root_url = "https://files.rcsb.org/download/"
    url = os.path.join(root_url,files[item]+".pdb")
    print(url)
    response = requests.get(url)
    open(files[item]+".pdb", "wb").write(response.content)
    print(url)

    # https://files.rcsb.org/download/2GHQ.pdb