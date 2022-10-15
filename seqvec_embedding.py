# Install bio_embeddings using the command: pip install bio-embeddings[all]

# from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
from bio_embeddings.embed import SeqVecEmbedder

import numpy as np
import torch 
import pandas as pd
import csv
# from proteins_to_graphs import ProteinDataset
'''
SeqVecEmbedder is ELMo-based language model but it applies on sequence of protein
It takes in sequence of a protein as input.

If we use SeqVecEmbedder to extract node features, it only has information within the protein

What if we include information beyond just a single protein? Is that possible? Maybe we can include variation of this protein to train the SeqVecEmbedder?

During the last meeting with Dr. Xu, he told me that biochemist compare proteins from different species to extract the amino acid feature
'''

# protein_seq = ProteinDataset("/Users/jiadonglou/Desktop/Benjamin/PPI_GNN/Human_features/")

# seq = 'MVTYDFGSDEMHD'
# #Potein sequence of length L

# embedder = SeqVecEmbedder()
# embedding = embedder.embed(seq)
# print("-=-=-=-=-=-=embedding-=-=-=-=-==-=-=")
# print(embedding)
# protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
# print("-=-=-=-=-=-=-=protein_embd-=-=-=-=-===-")
# print(protein_embd)
# np_arr = protein_embd.cpu().detach().numpy()
# print("-=-=-=-np_arr=-=-=-=-=-=")
# print(np_arr)


# # open the file in the write mode
# f = open('np_arr.csv', 'w')

# # create the csv writer
# writer = csv.writer(f)

# # write a row to the csv file
# writer.writerow(np_arr)

# # close the file
# f.close()

def embedding(seq):
    embedder = SeqVecEmbedder()
    #it returns the embedding of one sequence
    embedding = embedder.embed(seq)
    #convert the embedding to a tensor and sum them together
    protein_embd = torch.tensor(embedding).sum(dim=0)

    return protein_embd
    # Converting the tensor to a numpy array.
    # np_arr = protein_embd.cpu().detach().numpy
    # print(protein_embd)