import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

train_input = pd.read_csv('./lish-moa/train_features.csv')
train_output = pd.read_csv('./lish-moa/train_targets_scored.csv')

kf = KFold(7)

for a,b in kf.split(train_input):
    print(train_input.iloc[a])