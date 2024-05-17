import pickle
import os


# filepath = os.path.abspath(r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms')

# keys = 'keys_conf.pkl'
# # keys_file = os.path.join(filepath, keys)
# labels = 'labels_conf.pkl'
# # labels_file = os.path.join(filepath, labels)
# preds = 'preds_conf.pkl'
# # preds_file = os.path.join(filepath, preds)


import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def generate_confusion_matrices(keys_file, labels_file, preds_file):
    # Load the data from pickle files
        with open(keys_file, 'rb') as f:
            keys_list = pickle.load(f)
        with open(labels_file, 'rb') as f:
            labels_list = pickle.load(f)
        with open(preds_file, 'rb') as f:
            preds_list = pickle.load(f)
        
        truth = labels_list[0][0][1].item()
        predicted = preds_list[0][0][1].item()
        print('truth', truth) # Dit geeft the correcte index voor de true label in de keys_list deze waarde moet ingevuld worden bij de laatste 0
        print('predicted', predicted) # Zelfde als bij de labels maar dan voor the predicted label

        # Dit hieronder geeft the text labels terug: AzPos_xxx_ElPos_xxx.npz 
        true_label = list(keys_list[0][1].keys())[list(keys_list[0][1].values())[truth]]
        predicted_label = list(keys_list[0][1].keys())[list(keys_list[0][1].values())[predicted]]
        print('truth key thing', true_label) 
        print('predicted key thing', predicted_label) 

        # Works
        print(true_label==predicted_label)

        # Works
        print('ElPos_60' in predicted_label)



# Define file paths
filepath = os.path.abspath(r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms')
keys_file = os.path.join(filepath, 'keys_conf.pkl')
labels_file = os.path.join(filepath, 'labels_conf.pkl')
preds_file = os.path.join(filepath, 'preds_conf.pkl')

# Generate and print confusion matrices
matrices = generate_confusion_matrices(keys_file, labels_file, preds_file)
# for elpos, matrix in matrices.items():
#     print(f"Confusion Matrix for ElPos {elpos}:")
#     print(matrix)




# extra= 'extra_label.pkl'



# # DIT IS WAT ELK NUMMER NAAR REFERENCED QUA LABEL BETEKENIS
# with open(os.path.join(filepath, keys), 'rb') as fp:
#     keys = pickle.load(fp)
# # for i in range(len(keys[0])):
# #     print('key nr', i, ':', keys[0][i])
# print(keys)
# print(list(keys[0][1].keys())[list(keys[0][1].values())[0]])

# # DIT MOET IN DE RANGE VALLEN WANT DIT IS ACCURATE LABEL
# with open(os.path.join(filepath, labels), 'rb') as fp:
#     labels = pickle.load(fp)
# # for i in range(len(labels[0])):
# #     print('label nr', i, ':', labels[0][i]) 

# # DIT IS LOGISCH DAT FOUT IS DIT IS PREDICTED
# with open(os.path.join(filepath, preds), 'rb') as fp:
#     preds = pickle.load(fp)
# # for i in range(len(preds[0])):
# #     print('Pred nr', i, ':', preds[0][i])

# # These are the correct labels
# with open(os.path.join(filepath, extra), 'rb') as fp:
#     extra = pickle.load(fp)
# # for i in range(len(extra)):
# #     print('extra nr', i, ':', extra[i])