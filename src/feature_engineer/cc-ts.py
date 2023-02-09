#!/usr/bin/env python
# coding: utf-8

# Credit card balance time series feature extraction

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

# Scale data for NN training.
def scale_data(df_):
    df = df_.copy(deep=True)
    for f_ in df_.columns:
        if (df[f_].max() - df[f_].min() <= 10):
            df[f_] = df[f_] - df[f_].min()
            continue
        df[f_] = df[f_] - df[f_].median()
        scale = (df[f_].quantile(0.99) - df[f_].quantile(0.01))
        if scale == 0:
            scale = df[f_].max() - df[f_].min()
        df[f_] = df[f_] / scale
        if df[f_].max() > 10:
            rescale = df[f_] > df[f_].quantile(0.99)
            quantile99 = df[f_].quantile(0.99)
            quantile100 = df[f_].max()
            df[f_].loc[rescale] = quantile99 + (df[f_].loc[rescale] - quantile99) * (10 - quantile99) / (
                        quantile100 - quantile99)
        if df[f_].min() < -10:
            rescale = df[f_] < df[f_].quantile(0.01)
            quantile1 = df[f_].quantile(0.01)
            quantile0 = df[f_].min()
            df[f_].loc[rescale] = quantile1 + (df[f_].loc[rescale] - quantile1) * (-10 - quantile1) / (
                        quantile0 - quantile1)
        df[f_] = df[f_] - df[f_].min()
    return df


# Read credit card balance data and create features.

ccbl = pd.read_csv(f'{input_path}/credit_card_balance.csv')

ccbl = pd.concat([ccbl, pd.get_dummies(ccbl['NAME_CONTRACT_STATUS'], prefix='NAME_CONTRACT_STATUS')], axis=1)
del ccbl['NAME_CONTRACT_STATUS']

sum_feats = [f_ for f_ in ccbl.columns.values if
             ((f_.find('SK_ID_CURR') < 0) & (f_.find('MONTHS_BALANCE') < 0) & (f_.find('SK_ID_PREV') < 0))]
print('sum_feats', sum_feats)
sum_ccbl_mon = ccbl.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[sum_feats].sum()
sum_ccbl_mon['CNR_ACCOUNT_W_MONTH'] = ccbl.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])['SK_ID_PREV'].count()
ccbl = sum_ccbl_mon.reset_index()

# compute ratio after summing up account
ccbl['AMT_BALANCE_CREDIT_RATIO'] = (ccbl['AMT_BALANCE'] / (ccbl['AMT_CREDIT_LIMIT_ACTUAL'] + 0.001)).clip(-100, 100)
ccbl['AMT_CREDIT_USE_RATIO'] = (ccbl['AMT_DRAWINGS_CURRENT'] / (ccbl['AMT_CREDIT_LIMIT_ACTUAL'] + 0.001)).clip(-100,
                                                                                                               100)
ccbl['AMT_DRAWING_ATM_RATIO'] = ccbl['AMT_DRAWINGS_ATM_CURRENT'] / (ccbl['AMT_DRAWINGS_CURRENT'] + 0.001)
ccbl['AMT_DRAWINGS_OTHER_RATIO'] = ccbl['AMT_DRAWINGS_OTHER_CURRENT'] / (ccbl['AMT_DRAWINGS_CURRENT'] + 0.001)
ccbl['AMT_DRAWINGS_POS_RATIO'] = ccbl['AMT_DRAWINGS_POS_CURRENT'] / (ccbl['AMT_DRAWINGS_CURRENT'] + 0.001)
ccbl['AMT_PAY_USE_RATIO'] = ((ccbl['AMT_PAYMENT_TOTAL_CURRENT'] + 0.001) / (ccbl['AMT_DRAWINGS_CURRENT'] + 0.001)).clip(
    -100, 100)
ccbl['AMT_BALANCE_RECIVABLE_RATIO'] = ccbl['AMT_BALANCE'] / (ccbl['AMT_TOTAL_RECEIVABLE'] + 0.001)
ccbl['AMT_DRAWING_BALANCE_RATIO'] = ccbl['AMT_DRAWINGS_CURRENT'] / (ccbl['AMT_BALANCE'] + 0.001)
ccbl['AMT_RECEIVABLE_PRINCIPAL_DIFF'] = ccbl['AMT_TOTAL_RECEIVABLE'] - ccbl['AMT_RECEIVABLE_PRINCIPAL']
ccbl['AMT_PAY_INST_DIFF'] = ccbl['AMT_PAYMENT_CURRENT'] - ccbl['AMT_INST_MIN_REGULARITY']

rejected_features = ['AMT_RECIVABLE', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_DRAWINGS_ATM_CURRENT',
                     'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']
for f_ in rejected_features:
    del ccbl[f_]

ccbl.iloc[:, 3:] = scale_data(ccbl.iloc[:, 3:])

del sum_ccbl_mon
gc.collect()
ccbl.head()

# Read target from main table.

data_app = pd.read_csv(f'{input_path}/application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
data_test = pd.read_csv(f'{input_path}/application_test.csv', usecols=['SK_ID_CURR'])

trn_id = data_app['SK_ID_CURR'].loc[data_app.SK_ID_CURR.isin(ccbl.SK_ID_CURR)]
test_id = data_test['SK_ID_CURR'].loc[data_test['SK_ID_CURR'].isin(ccbl.SK_ID_CURR)]

# Split train and test set. Group by ID and month to create time series.

ccbl_trn = ccbl.loc[ccbl.SK_ID_CURR.isin(trn_id)]
ccbl_test = ccbl.loc[ccbl.SK_ID_CURR.isin(test_id)]
feats = ccbl.columns.values[2:]
ccbl_trn = ccbl_trn.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[feats].sum()
ccbl_test = ccbl_test.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[feats].sum()

# Convert dataframe to 3D array (n_sample * n_time_step * n_features) for GRU network training.

train_x = np.array(ccbl_trn.to_xarray().to_array())
train_x = train_x.swapaxes(0, 1).swapaxes(1, 2)

test_x = np.array(ccbl_test.to_xarray().to_array())
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
    model.add(GRU(16, input_shape=(
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
    model.compile(loss='binary_crossentropy', optimizer=Adam(decay=0.001))
    model.fit(trn_x, trn_y,
              validation_data=(val_x, val_y),
              epochs=20, batch_size=3000,
              class_weight={0: 1, 1: 10},
              callbacks=[ival], verbose=5)

    oof_preds[val_idx] = model.predict(val_x)[:, 0]
    sub_preds += model.predict(test_x)[:, 0] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del model, trn_x, trn_y, val_x, val_y
    gc.collect()

# Save model prediction to disk.

cc_score_train = pd.DataFrame({'cc_score': oof_preds}, index=trn_id)
cc_score_test = pd.DataFrame({'cc_score': sub_preds}, index=test_id)
cc_score_train.to_csv(f'{current_path}/../../output/cc_score_train.csv')
cc_score_test.to_csv(f'{current_path}/../../output/cc_score_test.csv')
