import pickle
import os

os.path.abspath(r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms\labels_after_training')
with open(r'C:\Users\arnou\Documents\Radboud\Thesis\slowfast18\LogMelSpectrograms\LogMelSpectrograms\labels_after_training.pkl', 'rb') as f:
    data = pickle.load(f)