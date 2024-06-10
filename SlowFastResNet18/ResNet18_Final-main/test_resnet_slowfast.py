# script to test resnet on independent dataset

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd 
import os  
import pickle

from Bases.resnet34_slow_stride import ResNet, BasicBlock

from utils_resnet_slowfast import Dataset
from test_utils_resnet_slowfast import predict

filepath = os.path.abspath(r'LogMelSpectrograms_TestSet\LogMelSpectrograms_TestSet')

# Read npz files to create list of IDs
file_IDs = [f.split('.')[0] for f in os.listdir(filepath) if f.endswith('.npz')]

# Read file to create list of IDs 
filename_keypairs = 'labels_after_training_slow.pkl'
filename_model = 'trained_slow_twochannel_50epochs_34.pt' 
filename_predictions = 'predictions_and_truelabels_slow.csv'


# Extract labels from filenames using the last 23 character, should be: AzPos_xxx_ElPos_xxx.npz
labels = [f[-23:] for f in os.listdir(filepath) if f.endswith('.npz')]

# Categorize files based on labels
classes = set(labels)
indices_dict = {class_: [i for i, label in enumerate(labels) if label == class_] for class_ in classes}

# Check to see if class names are correct
#print(classes)

# Should be equal to number of test cases
num_test = 25

# Split indices for training and validation
indices_test= []
for class_, indices in indices_dict.items():
    indices_test.extend(indices[:num_test])

# Create data and label splits
labels_test = {file_IDs[i]: labels[i] for i in indices_test}
partition = {'test': [file_IDs[i] for i in indices_test]}


# Read dictionary pkl file
with open(os.path.join(filepath,filename_keypairs), 'rb') as fp:
    keys = pickle.load(fp)
    # print(keys)

# Check if they match
# print("Sample keys from dictionary:", list(keys.keys())[:5])
# print("Sample extracted labels:", labels[:5])

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params_test = {'batch_size': 32,
              'shuffle': False}

# Define parameters.
nr_channels    = 2 # This specifies the number of channels that is used. Can be set either to 1 or 3.
nr_classes     = 114 

# Generators
test_set = Dataset(partition['test'], labels_test, filepath)
test_generator = torch.utils.data.DataLoader(test_set,**params_test)

criterion = nn.CrossEntropyLoss()


# Initialise model
model = ResNet(img_channels=nr_channels, num_layers=34, block=BasicBlock, num_classes=nr_classes).to(device)
# load state dict
model.load_state_dict(torch.load(os.path.join(filepath,filename_model)))

# main execution block
if __name__ == '__main__':
    # Perform prediction
    test_loss, test_acc, keys, final_label, final_preds, final_keys, extra_label_thing = predict(
            model, 
            test_generator, 
            criterion,
            device
        )

# Calculate accuracy
# evaluation_correct = (predictions == true_labels).sum().item()
# accuracy_evaluation = evaluation_correct
print(f"Test loss: {test_loss:.3f}, Test acc: {test_acc:.3f}")
# print('evaluation on independent dataset complete, accuracy = ' + str(accuracy_evaluation) + '%')
# print(true_labels)
# check whether key-value pairs match training key-value pairs
#print(keys)

# Save predictions and true labels for subsequent analysis
# predictions = predictions.cpu() # bring back to cpu for numpy
# true_labels = true_labels.cpu()
# df = pd.DataFrame({'predictions':predictions.numpy(),'true_labels':true_labels.numpy()}) #convert to a dataframe
# df.to_csv(os.path.join(filepath,filename_predictions), index={'Predictions','TrueLabels'}) #save to file