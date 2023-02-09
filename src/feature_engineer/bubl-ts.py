#!/usr/bin/env python
# coding: utf-8

# bureau balance time series feature extraction

# 数据读取
# 预处理和特征提取，转换成时间序列数据。
# 数据拆分，并将这些数据输入到GRU网络中进行训练。
# 最后，将预测结果保存下来，供最终训练使用


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import gc
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
import logging
import os
gc.enable()

current_path = os.path.dirname(__file__)
input_path = f'{current_path}/../../input'

# Read bureau balance data and create features.

buro = pd.read_csv(f'{input_path}/bureau.csv')
buro_id_map = buro.groupby('SK_ID_BUREAU')['SK_ID_CURR'].min()
buro.head()

# buro has 1716428 SK_ID_BUREAU
# bubl has 817395 SK_ID_BUREAU
# 942074 buro_id in buro not present in bubl
# 43041 buro_id in bubl not present in buro
bubl = pd.read_csv(f'{input_path}/bureau_balance.csv')
bubl['STATUS_COMPLETE'] = 0
bubl['STATUS_COMPLETE'].loc[bubl['STATUS'] == 'C'] = 1
bubl['STATUS_X'] = 0
bubl['STATUS_X'].loc[bubl['STATUS'] == 'X'] = 1
bubl['STATUS_DPD'] = -1
bubl['STATUS_DPD'].loc[bubl['STATUS'].isin(['0', '1', '2', '3', '4', '5'])] = bubl['STATUS']
bubl['STATUS_DPD'] = bubl['STATUS_DPD'].astype('int32')
bubl['SK_ID_CURR'] = bubl['SK_ID_BUREAU'].map(buro_id_map)
bubl = bubl.loc[bubl['SK_ID_CURR'].notna()]
bubl['SK_ID_CURR'] = bubl['SK_ID_CURR'].astype('int')
bubl.head()

# Read target from main table.

data_app = pd.read_csv(f'{input_path}/application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
data_test = pd.read_csv(f'{input_path}/application_test.csv', usecols=['SK_ID_CURR'])

trn_id = data_app['SK_ID_CURR'].loc[data_app.SK_ID_CURR.isin(bubl.SK_ID_CURR)]
test_id = data_test['SK_ID_CURR'].loc[data_test['SK_ID_CURR'].isin(bubl.SK_ID_CURR)]

# Split train and test set. Groupby ID and month to create time series.

bubl_trn = bubl.loc[bubl.SK_ID_CURR.isin(trn_id)]
bubl_test = bubl.loc[bubl.SK_ID_CURR.isin(test_id)]
num_aggregations = {
    'SK_ID_BUREAU': ['count'],
    'STATUS_COMPLETE': ['sum'],
    'STATUS_X': ['sum'],
    'STATUS_DPD': ['sum', 'mean', 'max'],
}
bubl_trn = bubl_trn.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg(num_aggregations)
bubl_test = bubl_test.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg(num_aggregations)
bubl_trn.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bubl_trn.columns.tolist()])
bubl_test.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bubl_test.columns.tolist()])
bubl_test.head()

# Convert dataframe to 3D array (n_sample * n_time_step * n_features) for GRU network training.

train_x = np.array(bubl_trn.to_xarray().to_array())
train_x = train_x.swapaxes(0, 1).swapaxes(1, 2)
test_x = np.array(bubl_test.to_xarray().to_array())
test_x = test_x.swapaxes(0, 1).swapaxes(1, 2)
train_x[np.isnan(train_x)] = -9
test_x[np.isnan(test_x)] = -9
train_y = data_app['TARGET'].loc[data_app.SK_ID_CURR.isin(trn_id)]
train_x.shape, test_x.shape, train_y.shape

# Define GRU model. Use callback to evaluate auc metric.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam

def build_model(time_step, n_features):
    model = Sequential()
    model.add(GRU(4, input_shape=(
    time_step, n_features)))  # unit: #of neurons in each LSTM cell? input_shape=(time_step, n_features)
    model.add(Dense(1, activation='sigmoid'))
    return model

class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == (self.interval - 1):
            y_pred = self.model.predict(self.X_val, verbose=0)[:, 0]
            score = roc_auc_score(self.y_val, y_pred)
            print('roc score', score)

# Training...

# Run a 5 fold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
oof_preds = np.zeros(train_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    trn_x, val_x = train_x[trn_idx], train_x[val_idx]
    trn_y, val_y = train_y.values[trn_idx], train_y.values[val_idx]
    ival = IntervalEvaluation(validation_data=(val_x, val_y), interval=5)

    model = build_model(trn_x.shape[1], trn_x.shape[2])
    model.compile(loss='binary_crossentropy', optimizer=Adam(decay=0.0002))
    model.fit(trn_x, trn_y,
              validation_data=(val_x, val_y),
              epochs=40, batch_size=5000,
              class_weight={0: 1, 1: 10},
              callbacks=[ival], verbose=5)

    oof_preds[val_idx] = model.predict(val_x)[:, 0]
    sub_preds += model.predict(test_x)[:, 0] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del model, trn_x, trn_y, val_x, val_y
    gc.collect()

# Save model prediction to disk.

bubl_score_train = pd.DataFrame({'bubl_score': oof_preds}, index=trn_id)
bubl_score_test = pd.DataFrame({'bubl_score': sub_preds}, index=test_id)
bubl_score_train.to_csv(f'{current_path}/../../output/bubl_score_train.csv')
bubl_score_test.to_csv(f'{current_path}/../../output/bubl_score_test.csv')
