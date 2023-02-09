#!/usr/bin/env python
# coding: utf-8

# Pos-cash balance time series feature extraction

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

import os
gc.enable()

current_path = os.path.dirname(__file__)
input_path = f'{current_path}/../../input'

# Read pos-cash balance and create features.

pos = pd.read_csv(f'{input_path}/POS_CASH_balance.csv')
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'], prefix='NAME_CONTRACT_STATUS')], axis=1)
pos['CNT_INSTALMENT'] /= 10
pos['CNT_INSTALMENT_FUTURE'] /= 10
del pos['NAME_CONTRACT_STATUS']

# Read target from main table.

data_app = pd.read_csv(f'{input_path}/application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
data_test = pd.read_csv(f'{input_path}/application_test.csv', usecols=['SK_ID_CURR'])

trn_id = data_app['SK_ID_CURR'].loc[data_app.SK_ID_CURR.isin(pos.SK_ID_CURR)]
test_id = data_test['SK_ID_CURR'].loc[data_test['SK_ID_CURR'].isin(pos.SK_ID_CURR)]

# Split train and test set. Group by ID and month to create time series.

pos_trn = pos.loc[pos.SK_ID_CURR.isin(trn_id)]
pos_test = pos.loc[pos.SK_ID_CURR.isin(test_id)]
num_aggregations = {
    'SK_ID_PREV': ['count'],
    'CNT_INSTALMENT': ['sum', 'max', 'mean'],
    'CNT_INSTALMENT_FUTURE': ['sum', 'max', 'mean'],
    'NAME_CONTRACT_STATUS_Approved': ['sum'],
    'NAME_CONTRACT_STATUS_Canceled': ['sum'],
    'NAME_CONTRACT_STATUS_Completed': ['sum'],
    'NAME_CONTRACT_STATUS_Demand': ['sum'],
    'NAME_CONTRACT_STATUS_Returned to the store': ['sum'],
    'NAME_CONTRACT_STATUS_Signed': ['sum'],
    'NAME_CONTRACT_STATUS_XNA': ['sum'],
    'SK_DPD': ['sum', 'mean'],
    'SK_DPD_DEF': ['sum', 'mean']
}
pos_trn = pos_trn.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg(num_aggregations)
pos_test = pos_test.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg(num_aggregations)
pos_trn.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_trn.columns.tolist()])
pos_test.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_test.columns.tolist()])
pos_test.head()

# Convert dataframe to 3D array (n_sample * n_time_step * n_features) for GRU network training.

train_x = np.array(pos_trn.to_xarray().to_array())
train_x = train_x.swapaxes(0, 1).swapaxes(1, 2)
test_x = np.array(pos_test.to_xarray().to_array())
test_x = test_x.swapaxes(0, 1).swapaxes(1, 2)
train_x[np.isnan(train_x)] = -9
test_x[np.isnan(test_x)] = -9
train_y = data_app['TARGET'].loc[data_app.SK_ID_CURR.isin(trn_id)]

# Define GRU model. Use callback to evaluate auc metric.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam


def build_model(time_step, n_features):
    model = Sequential()
    model.add(GRU(8, input_shape=(
    time_step, n_features)))  # unit: #of neurons in each LSTM cell? input_shape=(time_step, n_features)
    model.add(Dense(1, activation='sigmoid'))
    return model


from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
import logging


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
    model.compile(loss='binary_crossentropy', optimizer=Adam(decay=0.0005))
    model.fit(trn_x, trn_y,
              validation_data=(val_x, val_y),
              epochs=20, batch_size=5000,
              class_weight={0: 1, 1: 10},
              callbacks=[ival], verbose=5)

    oof_preds[val_idx] = model.predict(val_x)[:, 0]
    sub_preds += model.predict(test_x)[:, 0] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del model, trn_x, trn_y, val_x, val_y
    gc.collect()

# Save model prediction to disk.

pos_score_train = pd.DataFrame({'pos_score': oof_preds}, index=trn_id)
pos_score_test = pd.DataFrame({'pos_score': sub_preds}, index=test_id)
pos_score_train.to_csv(f'{current_path}/../../output/pos_score_train.csv')
pos_score_test.to_csv(f'{current_path}/../../output/pos_score_test.csv')
