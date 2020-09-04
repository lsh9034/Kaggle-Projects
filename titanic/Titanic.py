from keras import models, layers, regularizers, initializers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_grpah(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs,acc, 'bo', label='Training acc', markersize=1)
    plt.plot(epochs, val_acc, 'b', label='Validation acc', markersize=1)
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss', markersize=1)
    plt.plot(epochs, val_loss, 'b', label='Validation loss', markersize=1)
    plt.legend()
    plt.show()

def extract_title(df):
    title = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    return title

def map_title(df):
    title_category = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
    }
    new_title = df['Title'].map(title_category)
    return new_title

# Function to identify passengers who have the title 'Miss' and, 1 or 2 value in the 'Parch' column
def is_young(df):
    young = []
    for index, value in df['Parch'].items():
        if ((df.loc[index, 'Title'] == 'Miss') and (value == 1 or value == 2)):
            young.append(1)
        else:
            young.append(0)
    return young

def Replacing(df):
    df['Sex'] = df['Sex'] == 'male'
    df['Sex'] = df['Sex'].astype(int)

    #df['Is_Young(Miss)'] = is_young(df)
    #print(df.groupby(['Pclass', 'Title', 'Is_Young(Miss)'])['Age'].median())
    #group_age  = df.groupby(['Pclass', 'Title', 'Is_Young(Miss)']).median()['Age']

    #if isTest == 0: df['Age'].fillna(df.groupby(['Pclass', 'Title', 'Is_Young(Miss)'])['Age'].transform('median'),inplace=True)
    #else:
    #    print(df.groupby(['Pclass', 'Title', 'Is_Young(Miss)'])['Age'].median())
    #    df['Age'].fillna(df.groupby(['Pclass', 'Title', 'Is_Young(Miss)'])['Age'].transform('median'),inplace=True)
    #df['Age'].fillna(df.groupby('Sex')['Age'].transform('median'), inplace=True)

    #d = {'C': 1, 'Q': 2, 'S': 3}
    #df['Embarked'] = df['Embarked'].map(d)
    #df['Embarked'].fillna(0, inplace=True)

    df = pd.get_dummies(df, columns=['Embarked', 'Title'])

    #df['Name'] = df['Name'].transform(lambda x: len(x))
    #df['Cabin'].fillna('',inplace=True)
    #df['Cabin'] = df['Cabin'].transform(lambda x: len(x.split()))

    #df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    return df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

sizeOfTrain = 791
drop_colums = ['PassengerId','Name', 'Cabin', 'Ticket', 'Is_Young(Miss)']
normalize_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#Replacing
df['Title'] = extract_title(df)
df['Title'] = map_title(df)
df['Is_Young(Miss)'] = is_young(df)

test_df['Title'] = extract_title(test_df)
test_df['Title'] = map_title(test_df)
test_df['Is_Young(Miss)'] = is_young(test_df)


group_age = df.groupby(['Pclass', 'Title', 'Is_Young(Miss)']).median()['Age']
df.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)
df['Age'].fillna(group_age, inplace=True)
df.reset_index(drop=True, inplace=True)

test_df.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)
test_df['Age'].fillna(group_age,inplace=True)
test_df.reset_index(drop=True, inplace=True)

group_fare = df.groupby('Pclass').median()['Fare']
df.set_index('Pclass', drop=False, inplace=True)
df['Fare'].fillna(group_fare, inplace=True)
df.reset_index(drop=True, inplace=True)

df = Replacing(df)
test_df = Replacing(test_df)
#print(test_df)

# Dropping
df = df.drop(drop_colums, axis=1)
target_df = df['Survived']
df = df.drop(['Survived'], axis=1)

passengerId = list(test_df['PassengerId'])
test_df.drop(drop_colums,inplace=True,axis=1)

#print(df)
#print(test_df)
print(df.isnull().sum())
print(test_df.isnull().sum())

# Convert type
#np_data = np.array(df)
#np_target = np.array(target_df)
#np_test_data = np.array(test_df)

# Split Data
train_input = df[:sizeOfTrain]
train_target = target_df[:sizeOfTrain]
dev_input = df[sizeOfTrain:]
dev_target = target_df[sizeOfTrain:]

# Normalizing
mean = df[normalize_columns].mean(axis=0)
std = df[normalize_columns].std(axis=0)
train_input[normalize_columns] -= mean
train_input[normalize_columns] /= std
dev_input[normalize_columns] -= mean
dev_input[normalize_columns] /= std

test_df[normalize_columns] -= mean
test_df[normalize_columns] /= std

train_input[normalize_columns] += 2.2
dev_input[normalize_columns] += 2.2
test_df[normalize_columns] += 2.2
print(train_input)
print(dev_input)
print(test_df)

# Make model
model = models.Sequential()
model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.he_normal(), activation='relu',input_shape=(15,)))
'''
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
for i in range(1,5):
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.he_normal(), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

model.add(layers.Dense(1,activation='sigmoid'))
'''
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(train_input,train_target,
                    batch_size=64,
                    epochs=10000,
                    validation_data=(dev_input,dev_target),
                    verbose=2)
draw_grpah(history)

survived = model.predict(test_df,1).flatten()
print(survived)
for i in range(len(survived)):
    if survived[i]>=0.5: survived[i]=1
    else: survived[i]=0
survived = survived.astype(int)
d = {'PassengerId':passengerId, 'Survived':survived}
result = pd.DataFrame(data=d)
result.to_csv('result_titanic.csv', index=False)
