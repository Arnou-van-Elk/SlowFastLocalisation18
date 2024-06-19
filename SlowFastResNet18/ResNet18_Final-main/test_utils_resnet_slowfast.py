import torch
from tqdm import tqdm



def predict(model, test_generator, device, keys):
    model.eval()
    with torch.no_grad(): # Context-manager that disables gradient calculation. Useful for inference to reduce memory consumption.
        for i, data in tqdm(enumerate(test_generator), total=len(test_generator)):
            image, labels = data
            image = image.to(device)
            labels_unique = set(labels)
            keys = {key: value for key, value in zip(labels_unique,range(len(labels_unique)))}
            labels_int = torch.zeros(size=(len(labels),))
            for idx, label in enumerate(labels):
                labels_int[idx] = keys[labels[idx]]
            labels_int = labels_int.type(torch.LongTensor) # change data type for loss function
            labels = labels_int.to(device)
            # predict
            predicted_outputs = model(image)  
            _, preds = torch.max(predicted_outputs.data, 1)
            
        return preds, labels, keys
