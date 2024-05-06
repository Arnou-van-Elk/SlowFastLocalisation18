import torch
import wandb
from tqdm import tqdm

# Should be unaltered

# Training function.
def train(model, trainloader, optimizer, scheduler, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels_unique = set(labels)
        keys = {key: value for key, value in zip(labels_unique,range(len(labels_unique)))}
        labels_int = torch.zeros(size=(len(labels),))
        for idx, label in enumerate(labels):
            labels_int[idx] = keys[labels[idx]]
        labels_int = labels_int.type(torch.LongTensor) # change data type for loss function
        labels = labels_int.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    # learning rate scheduler
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    wandb.log({"lr_after": after_lr}) # monitor learning rate
    return epoch_loss, epoch_acc, keys

# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels_unique = set(labels)
            keys = {key: value for key, value in zip(labels_unique,range(len(labels_unique)))}
            labels_int = torch.zeros(size=(len(labels),))
            for idx, label in enumerate(labels):
                labels_int[idx] = keys[labels[idx]]
            labels_int = labels_int.type(torch.LongTensor) # change data type for loss function
            labels = labels_int.to(device)
            
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    

    
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, keys
