#!/usr/bin/env python
# coding: utf-8

# Installment payment time series feature extraction

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

# Helper functions.
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64"]]

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df

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


# Read installment data and create features.

inst = pd.read_csv(f'{input_path}/installments_payments.csv')
inst['DAYS_ENTRY_PAYMENT_weighted'] = inst['DAYS_ENTRY_PAYMENT'] * inst['AMT_PAYMENT']
inst = inst.groupby(['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER']).agg({
    'DAYS_INSTALMENT': 'mean',
    'DAYS_ENTRY_PAYMENT_weighted': 'sum',
    'AMT_INSTALMENT': 'mean',
    'AMT_PAYMENT': 'sum'})
inst['DAYS_ENTRY_PAYMENT'] = inst['DAYS_ENTRY_PAYMENT_weighted'] / inst['AMT_PAYMENT']
inst = inst.reset_index()
del inst['DAYS_ENTRY_PAYMENT_weighted']
inst['AMT_PAYMENT_PERC'] = inst['AMT_PAYMENT'] / (1 + inst['AMT_INSTALMENT'])
inst['DPD'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
inst['DBD'] = (inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']).clip(lower=0)
inst['MONTHS_BALANCE'] = (inst['DAYS_INSTALMENT'] / 30.4375).astype('int')
del inst['DAYS_ENTRY_PAYMENT'], inst['DAYS_INSTALMENT']
gc.collect()
# apply logarithm to make distribution more normal
inst['AMT_INSTALMENT_LOG'] = inst['AMT_INSTALMENT'].apply(np.log1p)
inst['AMT_PAYMENT_LOG'] = inst['AMT_PAYMENT'].apply(np.log1p)
inst[['AMT_INSTALMENT', 'AMT_PAYMENT']] = scale_data(inst[['AMT_INSTALMENT', 'AMT_PAYMENT']])

# Read target from main table.

data_app = pd.read_csv(f'{input_path}/application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
data_test = pd.read_csv(f'{input_path}/application_test.csv', usecols=['SK_ID_CURR'])

trn_id = data_app['SK_ID_CURR'].loc[data_app.SK_ID_CURR.isin(inst.SK_ID_CURR)]
test_id = data_test['SK_ID_CURR'].loc[data_test['SK_ID_CURR'].isin(inst.SK_ID_CURR)]
print(trn_id.shape, test_id.shape)

# Split train and test set. Group by ID and month to create time series.

inst_trn = inst.loc[inst.SK_ID_CURR.isin(trn_id)]
inst_test = inst.loc[inst.SK_ID_CURR.isin(test_id)]
num_aggregations = {
    'SK_ID_PREV': ['count'],
    'NUM_INSTALMENT_NUMBER': ['sum', 'max'],
    'AMT_INSTALMENT': ['sum', 'mean'],
    'AMT_PAYMENT': ['sum', 'mean'],
    'AMT_PAYMENT_PERC': ['mean', 'max'],
    'DPD': ['sum', 'max', 'mean'],
    'DBD': ['sum', 'max', 'mean'],
    'AMT_INSTALMENT_LOG': ['mean'],
    'AMT_PAYMENT_LOG': ['mean']
}
inst_trn = inst_trn.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg(num_aggregations)
inst_test = inst_test.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg(num_aggregations)
inst_trn.columns = pd.Index([e[0] + "_" + e[1].upper() for e in inst_trn.columns.tolist()])
inst_test.columns = pd.Index([e[0] + "_" + e[1].upper() for e in inst_test.columns.tolist()])

inst_trn = downcast_dtypes(inst_trn)
inst_test = downcast_dtypes(inst_test)
del inst
gc.collect()

# Convert dataframe to 3D array (n_sample * n_time_step * n_features) for GRU network training.

train_x = np.array(inst_trn.to_xarray().to_array())
train_x = train_x.swapaxes(0, 1).swapaxes(1, 2)
test_x = np.array(inst_test.to_xarray().to_array())
test_x = test_x.swapaxes(0, 1).swapaxes(1, 2)
train_x[np.isnan(train_x)] = -9
test_x[np.isnan(test_x)] = -9
train_y = data_app['TARGET'].loc[data_app.SK_ID_CURR.isin(trn_id)]

del inst_trn, inst_test
gc.collect()

print(train_x.shape, test_x.shape, train_y.shape)

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


print('Training...')

# Run a 5 fold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
oof_preds = np.zeros(train_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    trn_x, val_x = train_x[trn_idx], train_x[val_idx]
    trn_y, val_y = train_y.values[trn_idx], train_y.values[val_idx]
    ival = IntervalEvaluation(validation_data=(val_x, val_y), interval=5)

    model = build_model(trn_x.shape[1], trn_x.shape[2])
    model.compile(loss='binary_crossentropy', optimizer=Adam(decay=0.0001))
    model.fit(trn_x, trn_y,
              validation_data=(val_x, val_y),
              epochs=40, batch_size=8000,
              class_weight={0: 1, 1: 10},
              callbacks=[ival], verbose=5)

    oof_preds[val_idx] = model.predict(val_x)[:, 0]
    sub_preds += model.predict(test_x)[:, 0] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del model, trn_x, trn_y, val_x, val_y
    gc.collect()

# Save model prediction to disk.
inst_score_train = pd.DataFrame({'inst_score': oof_preds}, index=trn_id)
inst_score_test = pd.DataFrame({'inst_score': sub_preds}, index=test_id)
inst_score_train.to_csv(f'{current_path}/../../output/inst_score_train.csv')
inst_score_test.to_csv(f'{current_path}/../../output/inst_score_test.csv')