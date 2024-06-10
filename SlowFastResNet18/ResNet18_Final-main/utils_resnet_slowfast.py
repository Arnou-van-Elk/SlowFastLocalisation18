import torch
import os
import numpy as np

# Adjusted version for LogMelSpectograms .npz files
class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, filepath):
        self.labels = labels
        self.list_IDs = list_IDs
        self.filepath = filepath  # Store filepath as an instance variable

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generate one sample of data
        ID = self.list_IDs[index]
        
        # Full path to the file
        file_path = os.path.join(self.filepath, ID + '.npz')
        
        # Load data from .npz file
        data = np.load(file_path)['arr_0'].astype(np.float32)
        
        # Get label for the sample
        y = self.labels[ID]

        return data, y
