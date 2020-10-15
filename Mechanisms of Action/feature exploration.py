import numpy as np  # linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
def Replacing(df):
    df['cp_type'] = pd.get_dummies(df['cp_type'], drop_first=True)

    dic  = {24: 0.5, 48: 1, 72: 1.5}
    df['cp_time'] = df['cp_time'].map(dic)

    #$df = pd.get_dummies(df, columns=['cp_time'])

    df['cp_dose'] = pd.get_dummies(df['cp_dose'], drop_first=True)

    return df

train_input = pd.read_csv('./lish-moa/train_features.csv')
train_output = pd.read_csv('./lish-moa/train_targets_scored.csv')
test_input = pd.read_csv('./lish-moa/test_features.csv')

train_input  = Replacing(train_input)

drop_columns = ['sig_id']
train_input = train_input.drop(drop_columns, axis=1)
train_output = train_output.drop(drop_columns, axis=1)
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X,y)
print(clf.feature_importances_)