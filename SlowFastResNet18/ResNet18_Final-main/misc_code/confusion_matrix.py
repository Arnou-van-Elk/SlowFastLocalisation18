import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re


def extract_azpos(text):
    # Regex pattern to capture digits immediately following 'AzPos_'
    pattern = r'AzPos_(\d+)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the captured digits
    if match:
        return match.group(1)  # group(1) refers to the first captured group which is the digits after 'AzPos_'
    else:
        return None


# HOOGSTWAARSCHIJNLIJK EXTRA [50] of [49] NODIG BIJ DE LISTS
def gen_tuple_list(keys_file, labels_file, preds_file):
    with open(keys_file, 'rb') as f:
        keys_list = pickle.load(f)

    with open(labels_file, 'rb') as f:
        labels_list = pickle.load(f)

    with open(preds_file, 'rb') as f:
        preds_list = pickle.load(f)

    categories = ['ElPos_000', 'ElPos_20', 'ElPos_45', 'ElPos_60', 'ElPos_-20', 'ElPos_-45']
    category_lists = {key: [] for key in categories}
    
    for j in range(len(labels_list[0])-2):
        for i in range(len(labels_list[0][j])-1):
            truth = labels_list[0][j][i].item()
            predicted = preds_list[0][j][i].item()

            values_list = list(keys_list[0][j].values())
            keys_list_j = list(keys_list[0][j].keys())

            if predicted < len(values_list) and truth < len(values_list):
                predicted_label = keys_list_j[values_list[predicted]]
                true_label = keys_list_j[values_list[truth]]

                for category in categories:
                    if category in true_label:
                        category_lists[category].append((extract_azpos(predicted_label), extract_azpos(true_label)))
                        break

    return tuple(category_lists[cat] for cat in categories)


def create_confusion_matrix(label_pairs, matrix_name):
    """
    Generate a confusion matrix from a list of tuples containing predicted and true labels.

    Args:
        label_pairs (list of tuples): Each tuple contains (predicted_label, true_label)

    Returns:
        np.array: A confusion matrix where the x-axis is true labels and the y-axis is predicted labels.
    """

    # Extract predicted and true labels
    predicted_labels = [pair[0] for pair in label_pairs]
    true_labels = [pair[1] for pair in label_pairs]

    # Define custom order for labels
    custom_order = ['270', '280', '290', '300', '310', '320', '330', '340', '350',
                    '000', '010', '020', '030', '040', '050', '060', '070', '080', '090']

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=custom_order)

    # Calculate the percentage of each entry relative to the column total
    column_sums = cm.sum(axis=0)
    column_sums = np.where(column_sums == 0, 1, column_sums)  # Avoid division by zero
    cm_percentage = (cm / column_sums) * 100

    # Adjusting each column to sum exactly to 100%
    for i in range(cm_percentage.shape[1]):  # Iterate over columns
        col_sum = np.sum(cm_percentage[:, i])
        correction_factor = 100 / col_sum
        cm_percentage[:, i] *= correction_factor
        cm_percentage[:, i] = np.round(cm_percentage[:, i], 0)  # Round to full integers

        # Correct any minor discrepancies caused by rounding
        correction = 100 - np.sum(cm_percentage[:, i])
        max_index = np.argmax(cm_percentage[:, i])
        cm_percentage[max_index, i] += correction

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, cmap='Blues', xticklabels=custom_order, yticklabels=custom_order, vmin=0, vmax=100)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title(matrix_name)
    plt.show()

    return cm_percentage


filepath = r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms'
keys_file = os.path.join(filepath, 'keys_conf_fast_34_lr.pkl')
labels_file = os.path.join(filepath, 'labels_conf_fast_34_lr.pkl')
preds_file = os.path.join(filepath, 'preds_conf_fast_34_lr.pkl')

ElPos_000, ElPos_20, ElPos_45, ElPos_60, ElPos_min20, ElPos_min45 = gen_tuple_list(keys_file, labels_file, preds_file)

ElPos_000_conf = create_confusion_matrix(ElPos_000, 'Fast stream confusion matrix for Elevation 000')
ElPos_20_conf = create_confusion_matrix(ElPos_20, 'Fast stream confusion matrix for Elevation 20')
ElPos_45_conf = create_confusion_matrix(ElPos_45, 'Fast stream confusion matrix for Elevation 45')
ElPos_60_conf = create_confusion_matrix(ElPos_60, 'Fast stream confusion matrix for Elevation 60')
ElPos_min20_conf = create_confusion_matrix(ElPos_min20, 'Fast stream confusion matrix for Elavation -20')
ElPos_min45_conf = create_confusion_matrix(ElPos_min45, 'Fast stream confusion matrix for Elavation -45')



## TESTING ZONE

# def gen_tuple_list(keys_file, labels_file, preds_file):

# # Load the data from pickle files
#     with open(keys_file, 'rb') as f:
#         keys_list = pickle.load(f)

#     with open(labels_file, 'rb') as f:
#         labels_list = pickle.load(f)

#     with open(preds_file, 'rb') as f:
#         preds_list = pickle.load(f)
#     ElPos_000, ElPos_20, ElPos_45, ElPos_60, ElPos_min20, ElPos_min45 = [], [], [], [],[],[]
#     for j in range(len(labels_list[0])-2):
#         for i in range(len(labels_list[0][j])-1):
#             truth = labels_list[0][j][i].item()
#             predicted = preds_list[0][j][i].item()
#             true_label = list(keys_list[0][j].keys())[list(keys_list[0][j].values())[truth]]
#             predicted_label = list(keys_list[0][j].keys())[list(keys_list[0][j].values())[predicted]]
#             if('ElPos_000' in true_label):
#                 ElPos_000.append((extract_azpos(predicted_label), extract_azpos(true_label)))
#             elif ('ElPos_20' in true_label):
#                 ElPos_20.append((extract_azpos(predicted_label), extract_azpos(true_label)))
#             elif('ElPos_45' in true_label):
#                 ElPos_45.append((extract_azpos(predicted_label), extract_azpos(true_label)))
#             elif('ElPos_60' in true_label):
#                 ElPos_60.append((extract_azpos(predicted_label), extract_azpos(true_label)))
#             elif('ElPos_-20' in true_label):
#                 ElPos_min20.append((extract_azpos(predicted_label), extract_azpos(true_label)))
#             elif('ElPos_-45' in true_label):
#                 ElPos_min45.append((extract_azpos(predicted_label), extract_azpos(true_label)))
#     return ElPos_000, ElPos_20, ElPos_45, ElPos_60, ElPos_min20, ElPos_min45
# Define file paths and generate matrices

# filepath = os.path.abspath(r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms')

# keys = 'keys_conf.pkl'
# # keys_file = os.path.join(filepath, keys)
# labels = 'labels_conf.pkl'
# # labels_file = os.path.join(filepath, labels)
# preds = 'preds_conf.pkl'
# # preds_file = os.path.join(filepath, preds)
# extra= 'extra_label.pkl'



# # DIT IS WAT ELK NUMMER NAAR REFERENCED QUA LABEL BETEKENIS
# with open(os.path.join(filepath, keys), 'rb') as fp:
#     keys = pickle.load(fp)
# # for i in range(len(keys[0])):
# #     print('key nr', i, ':', keys[0][i])

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

# print(len(labels[49]))