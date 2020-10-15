#import sys
#sys.path.append('../input/iterative-stratification/iterative-stratification-master') # Only use when you submit to kaggle
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import layers,models,regularizers,initializers,metrics,losses,Input,backend,optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import tensorflow_addons as tfa
from keras.utils import plot_model
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def draw_grpah(history):
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    #plt.plot(epochs, acc, 'bo', label='Training acc', markersize=1)
    #plt.plot(epochs, val_acc, 'b', label='Validation acc', markersize=1)
    #plt.legend()
    #plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss', markersize=1)
    plt.plot(epochs, val_loss, 'b', label='Validation loss', markersize=1)
    plt.legend()
    plt.show()


def Replacing(df):
    df['cp_type'] = df['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    #df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})

    #$df = pd.get_dummies(df, columns=['cp_time'])

    df['cp_dose'] = pd.get_dummies(df['cp_dose'], drop_first=True)

    return df

def StandardModel(features_size):
    print(features_size)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(features_size),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(400, activation='relu')),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(400, activation='relu')),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))
    ])
    model.compile(optimizer=tfa.optimizers.AdamW(lr=1e-3, weight_decay= 1e-5, clipvalue=900),loss=losses.BinaryCrossentropy(label_smoothing=0.001),metrics=logloss)
    #model.compile(optimizer=optimizers.Adam(learning_rate=15E-4), loss=losses.BinaryCrossentropy(),
     #             metrics=logloss)
    return model

def Sihyun_package(input_tensor, num_neurons, drop_rate):
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_neurons, kernel_initializer=tf.keras.initializers.he_normal(),
                                                             kernel_regularizer=tf.keras.regularizers.L1(0.01)))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)
    return x

def Updated_SimpleModel(features_size):
    model = models.Sequential()
    model.add(layers.Dense(2048, input_shape=(features_size,)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(206, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(learning_rate=1E-3), loss=losses.BinaryCrossentropy(label_smoothing=0.001),
                  metrics=logloss)
    return model

def SeperateModel(g_features, c_features):
    g_input = layers.Input(shape=(g_features,))
    g_x = Sihyun_package(g_input,2048, 0.4)
    g_x = layers.Activation('relu')(g_x)
    g_x = Sihyun_package(g_x,1024, 0.4)
    g_x = layers.Activation('relu')(g_x)
    g_output_tensor = layers.Dense(512, activation='relu')(g_x)

    c_input = layers.Input(shape=(c_features,))
    c_x = Sihyun_package(c_input, 128, 0.3)
    c_x = layers.Activation('relu')(c_x)
    c_output_tensor = layers.Dense(64, activation='relu')(c_x)

    merge_tensor = layers.concatenate([g_output_tensor, c_output_tensor], axis=-1)
    merge_x = Sihyun_package(merge_tensor, 512, 0.4)
    merge_x = layers.Activation('relu')(merge_x)
    output_tensor = layers.Dense(206, activation='sigmoid')(merge_x)

    model = models.Model([g_input,c_input], output_tensor)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def SeperateWeightNomalModel(g_features, c_features):
    g_input = tf.keras.layers.Input(shape=(g_features,))
    g_x = Sihyun_package(g_input, 2048, 0.4)
    g_x = tf.keras.layers.Activation('relu')(g_x)
    g_x = Sihyun_package(g_x, 1024, 0.4)
    g_x = tf.keras.layers.Activation('relu')(g_x)
    g_output_tensor = tfa.layers.WeightNormalization(tf.keras.layers.Dense(512, activation='relu'))(g_x)

    c_input = tf.keras.layers.Input(shape=(c_features,))
    c_x = Sihyun_package(c_input, 128, 0.3)
    c_x = tf.keras.layers.Activation('relu')(c_x)
    c_output_tensor = tfa.layers.WeightNormalization(tf.keras.layers.Dense(64, activation='relu'))(c_x)

    merge_tensor = tf.keras.layers.concatenate([g_output_tensor, c_output_tensor], axis=-1)
    merge_x = Sihyun_package(merge_tensor, 512, 0.4)
    merge_x = tf.keras.layers.Activation('relu')(merge_x)
    output_tensor = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation='sigmoid'))(merge_x)

    model = tf.keras.models.Model([g_input, c_input], output_tensor)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def Logistic(features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(206,activation='sigmoid',input_shape=(features,)))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

p_min = 0.001
p_max = 0.999

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,p_min,p_max)
    return -backend.mean(y_true*backend.log(y_pred) + (1-y_true)*backend.log(1-y_pred))


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# np.set_printoptions(threshold=np.inf)
'''
tf.enable_eager_execution()
label = [[0,0,0,1]]
inP = [[0,0,0,0.8]]
m = tf.keras.metrics.Accuracy()
m.update_state(label, inP)
print(m.result().numpy())
'''
# This is made by sihyun's feature selection.
drop_columns = ['sig_id']

train_input = pd.read_csv('./lish-moa/train_features.csv')
train_output = pd.read_csv('./lish-moa/train_targets_scored.csv')
test_input = pd.read_csv('./lish-moa/test_features.csv')
print('Done Input')
test_id = test_input['sig_id']
test_columns = train_output.columns[1:]

# Drop ctl_vehicle type
idx_ctl=[i for i in range(len(train_input['cp_type'])) if train_input['cp_type'][i]=='ctl_vehicle']
print(len(idx_ctl))
train_input.drop(idx_ctl, inplace=True)
train_output.drop(idx_ctl, inplace=True)
print(train_input)

# Don't drop test df
test_idx_ctl = [i for i in range(len(test_input['cp_type'])) if test_input['cp_type'][i] == 'ctl_vehicle']

# Dropping columns
train_input = train_input.drop(drop_columns, axis=1)
test_input = test_input.drop(drop_columns, axis=1)
train_output = train_output.drop(['sig_id'], axis=1)

# Replacing
train_input = Replacing(train_input)
test_input = Replacing(test_input)

print(train_input)

# K Fold CrossValidation
models1 = []
n_folds = 5
n_seeds = 6
oof=0
y_pred=0

np.random.seed(1)
seeds = np.random.randint(0,100,size=n_seeds)

kf = KFold(5)
K = 1
mi=1
final_model=None
Sum = 0
checkpoint_path='best_model.hdf5'
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        min_lr=1e-5,
        patience=5,
        verbose=1,
        mode='min',
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        restore_best_weights=True,
        patience=10,
        verbose=1,
    )
]

for seed in seeds:
    mlskf = MultilabelStratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    for train_index, dev_index in mlskf.split(train_input, train_output):
        k_fold_train_input = train_input.iloc[train_index]
        k_fold_train_output = train_output.iloc[train_index]
        k_fold_dev_input = train_input.iloc[dev_index]
        k_fold_dev_output = train_output.iloc[dev_index]
        print(len(train_input), len(train_index), len(dev_index))
        '''
        model = SeperateWeightNomalModel(774, 100)
        tf.keras.utils.plot_model(model, to_file='model.png')
        history = model.fit([k_fold_train_input.iloc[:,:774], k_fold_train_input.iloc[:, 774:]],k_fold_train_output,
                            batch_size=128,
                            epochs=35,
                            verbose=2,
                            callbacks=callbacks_list,
                            validation_data=([k_fold_dev_input.iloc[:,:774], k_fold_dev_input.iloc[:, 774:]], k_fold_dev_output))
        '''
        model = StandardModel(len(k_fold_train_input.columns))
        history = model.fit(k_fold_train_input, k_fold_train_output,
                            batch_size=128,
                            epochs=100,
                            verbose=2,
                            callbacks=callbacks_list,
                            validation_data=(k_fold_dev_input, k_fold_dev_output))

        model.load_weights(checkpoint_path)

        y_val = model.predict(k_fold_dev_input)
        oof += logloss(tf.constant(k_fold_dev_output, dtype=tf.float32), tf.constant(y_val, dtype=tf.float32)) / (
            n_folds * n_seeds)
        y_pred += model.predict(test_input) / (n_folds * n_seeds)
        '''
        # val_loss = model.evaluate([k_fold_dev_input.iloc[:,:774], k_fold_dev_input.iloc[:, 774:]], k_fold_dev_output)
        val_loss = model.evaluate(k_fold_dev_input, k_fold_dev_output)
        print(K, val_loss)
        # draw_grpah(history)
        if mi > val_loss:
            final_model = model

        models.append(model)
        Sum += val_loss
        K += 1
        '''
print('OOF score is', oof)

# Predict & Save
'''
#result = models[0].predict([test_input.iloc[:,:774], test_input.iloc[:, 774:]])

result = models[0].predict(test_input)
for i in range(1,len(models)):
    result+=models[i].predict(test_input)
    #result+=models[i].predict([test_input.iloc[:,:774], test_input.iloc[:, 774:]])
result /= len(models)

#result = final_model.predict(test_input)
print(result)
'''
result = np.clip(y_pred, p_min, p_max)
result[test_idx_ctl] = np.zeros(len(result[0]))
result = result.astype(float)
sig_id = pd.DataFrame(data=test_id, columns=['sig_id'])
result_data = pd.DataFrame(data=result, columns=test_columns)
result_csv = pd.concat([sig_id, result_data], axis=1)
result_csv.to_csv('submission.csv', index=False)