from keras import models, layers, regularizers, initializers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_grpah(history):
    acc = history.history['acc']
   # val_acc = history.history['val_acc']
    loss = history.history['loss']
   # val_loss = history.history['val_loss']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs,acc, 'bo', label='Training acc', markersize=1)
   # plt.plot(epochs, val_acc, 'b', label='Validation acc', markersize=1)
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss', markersize=1)
   # plt.plot(epochs, val_loss, 'b', label='Validation loss', markersize=1)
    plt.legend()
    plt.show()


def Age_1(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

def Replacing(df):

    df['Age'] = df[['Age','Pclass']].apply(Age_1,axis=1)

    df['Sex'] = pd.get_dummies(df['Sex'],drop_first=True)

    '''
    d = {'C': 1, 'Q': 2, 'S': 3}
    df['Embarked'] = df['Embarked'].map(d)
    df['Embarked'].fillna(0, inplace=True)
    '''
    df = pd.get_dummies(df,columns=['Embarked'])
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i>0 else 0)

    df['Name'] = df['Name'].transform(lambda x: len(x))
    df['Cabin'].fillna('', inplace=True)
    df['Cabin'] = df['Cabin'].transform(lambda x: len(x.split()))

    return df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

sizeOfTrain = 891
drop_colums = ['PassengerId','Ticket']
normalize_columns = ['Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#Replacing
df = Replacing(df)
test_df = Replacing(test_df)

# Dropping
df = df.drop(drop_colums, axis=1)
target_df = df['Survived']
df = df.drop(['Survived'], axis=1)

passengerId = list(test_df['PassengerId'])
test_df.drop(drop_colums,inplace=True,axis=1)

# Split Data
train_input = df.iloc[:sizeOfTrain]
train_target = target_df.iloc[:sizeOfTrain]
dev_input = df.iloc[sizeOfTrain:]
dev_target = target_df.iloc[sizeOfTrain:]
print(test_df.isnull().sum())

# Normalizing
mean = df[normalize_columns].mean(axis=0) #expediency way
std = df[normalize_columns].std(axis=0)
train_input[normalize_columns] -= mean
train_input[normalize_columns] /= std
dev_input[normalize_columns] -= mean
dev_input[normalize_columns] /= std

test_mean = test_df[normalize_columns].mean(axis=0)
test_std = test_df[normalize_columns].std(axis=0)
test_df[normalize_columns] -= test_mean
test_df[normalize_columns] /= test_std

train_input[normalize_columns]+=2.2
dev_input[normalize_columns]+=2.2
test_df[normalize_columns]+=2.2

print(train_input)
print(dev_input)
print(test_df)

# Make model
model = models.Sequential()
model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.he_normal(), activation='relu',input_shape=(11,)))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
for i in range(1,20):
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.he_normal(), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(train_input,train_target,
                    batch_size=64,
                    epochs=10000,
                    #validation_data=(dev_input,dev_target),
                    verbose=2)
draw_grpah(history)

survived = model.predict(test_df).flatten()
print(survived)
for i in range(len(survived)):
    if survived[i]>=0.5: survived[i]=1
    else: survived[i]=0
survived = survived.astype(int)
d = {'PassengerId':passengerId, 'Survived':survived}
result = pd.DataFrame(data=d)
result.to_csv('result_titanic.csv', index=False)
