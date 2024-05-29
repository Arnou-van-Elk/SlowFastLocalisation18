import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import wandb
import pickle
from torchvision import models  # Import torchvision models

from utils_resnet_slowfast import Dataset
from training_utils_resnet_slowfast import train, validate

# New filepath
filepath = os.path.abspath(r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms')

# Read LogMelspectograms files files to create list of IDs
file_IDs = [f.split('.')[0] for f in os.listdir(filepath) if f.endswith('.npz')]

# Extract labels from filenames using the last 23 character, should be: AzPos_xxx_ElPos_xxx.npz
labels = [f[-23:] for f in os.listdir(filepath) if f.endswith('.npz')]

# Categorize files based on labels
classes = set(labels)
indices_dict = {class_: [i for i, label in enumerate(labels) if label == class_] for class_ in classes}

# Some number to indicate the test and validation splits
num_train = 400
num_val = 100

indices_train, indices_val = [], []

# Split indices for training and validation
for class_, indices in indices_dict.items():
    indices_train.extend(indices[:num_train])
    indices_val.extend(indices[num_train:num_train + num_val])

# Create data and label splits
labels_train = {file_IDs[i]: labels[i] for i in indices_train}
labels_val = {file_IDs[i]: labels[i] for i in indices_val}
partition = {'train': [file_IDs[i] for i in indices_train], 'validation': [file_IDs[i] for i in indices_val]}

# Initialize WandB
wandb.init(project="Thesis")
wandb.run.name = 'Baseline_34'

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Define learning and training parameters.
batsize = 32
learning_rate = 0.0002
max_epochs = 50
nr_channels = 2
nr_classes = 114  # The amount of possible locations

# track hyperparameters and run metadata
wandb.config = {
    "learning_rate": learning_rate,
    "batch_size": batsize,
    "classes": nr_classes,
    "architecture": "ResNet-34",
    "dataset": "LogMelSpectograms",
    "epochs": max_epochs,
}

# Define parameters for data loaders
params_train = {'batch_size': batsize, 'shuffle': True}
params_val = {'batch_size': batsize, 'shuffle': True}

# Data generators
training_set = Dataset(partition['train'], labels_train, filepath)
training_generator = torch.utils.data.DataLoader(training_set, **params_train)

validation_set = Dataset(partition['validation'], labels_val, filepath)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)

# Model
model = models.resnet34(weights=False)  # Use torchvision's ResNet34
model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust the first conv layer to accept 2 channels
model.fc = nn.Linear(model.fc.in_features, nr_classes)  # Adjust the fully connected layer
model = model.to(device)

# Optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.75)

# Loss function
criterion = nn.CrossEntropyLoss()

# Main execution block
if __name__ == '__main__':
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    matrix_labels = []
    matrix_preds = []
    matrix_keys = []
    # Start the training.
    for epoch in range(max_epochs):
        print(f"[INFO]: Epoch {epoch+1} of {max_epochs}")
        train_epoch_loss, train_epoch_acc, keys_train = train(
            model, 
            training_generator, 
            optimizer, 
            scheduler,
            criterion,
            device
        )

        valid_epoch_loss, valid_epoch_acc, keys, final_label, final_preds, final_keys, extra_label_thing = validate(
            model, 
            validation_generator, 
            criterion,
            device
        )

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        wandb.log({
          "Epoch": epoch,
          "Train Loss": train_epoch_loss,
          "Train Acc": train_epoch_acc,
          "Valid Loss": valid_epoch_loss,
          "Valid Acc": valid_epoch_acc})

    matrix_keys.append(final_keys)
    matrix_labels.append(final_label)
    matrix_preds.append(final_preds)
    print('TRAINING COMPLETE')

    with open(os.path.join(filepath,'keys_conf_slowfast_34.pkl'), 'wb') as fp:
        pickle.dump(matrix_keys, fp)
        print('keysss saved successfully to file')

    with open(os.path.join(filepath,'labels_conf_slowfast_34.pkl'), 'wb') as fp:
        pickle.dump(matrix_labels, fp)
        print('labelsss saved successfully to file')

    with open(os.path.join(filepath,'preds_conf_slowfast_34.pkl'), 'wb') as fp:
        pickle.dump(matrix_preds, fp)
        print('predsss saved successfully to file')

    with open(os.path.join(filepath,'extra_label_slowfast_34.pkl'), 'wb') as fp:
        pickle.dump(extra_label_thing, fp)
        print('extrasss saved successfully to file')

    # save key-value dictionary for evaluation on independent dataset
    with open(os.path.join(filepath,'labels_after_training_conf.pkl'), 'wb') as fp:
        pickle.dump(keys_train, fp)
        print('dictionary saved successfully to file')

# save model weights after training
torch.save(model.state_dict(), os.path.join(filepath,'trained_model_twochannel_50epochs_34.pt'))