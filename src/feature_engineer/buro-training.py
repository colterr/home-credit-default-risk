#!/usr/bin/env python
# coding: utf-8

# bureau record training

# 这部分代码尝试利用历史bureau的记录与违约率之间的相关性。
# 每个bureau记录作为一条训练样本，标签是对应的当前标签 - 属于同一客户的bureau记录都将使用该客户的当前标签。
# 使用LightGBM分类器来预测每条bureau记录属于当前有违约贷款的客户的概率。
# 模型训练完成后，按照当前客户ID聚合分组，对预测结果进行进行统计，计算出均值/总和等统计信息。
# 将这些统计信息保存到磁盘中，以便后续和主表合并。

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from lightgbm import LGBMClassifier, LGBMRegressor
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import os

gc.enable()

current_path = os.path.dirname(__file__)
input_path = f'{current_path}/../../input'


data = pd.read_csv(f'{input_path}/application_train.csv', usecols=['SK_ID_CURR','TARGET'])


# Create features for each bureau record. Apart from raw features from bureau table, we also compute:
# * ratio between credit in debt and total credit
# * ratio between credit limit and total credit
# * ratio between credit overdue and total credit
# * difference between actual and expected account close date
#
# ...

buro = pd.read_csv(f'{input_path}/bureau.csv')

buro['DAYS_CREDIT_ENDDATE'].loc[buro['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
buro['DAYS_CREDIT_UPDATE'].loc[buro['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
buro['DAYS_ENDDATE_FACT'].loc[buro['DAYS_ENDDATE_FACT'] < -40000] = np.nan

buro['AMT_DEBT_RATIO'] = buro['AMT_CREDIT_SUM_DEBT']/(1+buro['AMT_CREDIT_SUM'])
buro['AMT_LIMIT_RATIO'] = buro['AMT_CREDIT_SUM_LIMIT']/(1+buro['AMT_CREDIT_SUM'])
buro['AMT_SUM_OVERDUE_RATIO'] = buro['AMT_CREDIT_SUM_OVERDUE']/(1+buro['AMT_CREDIT_SUM'])
buro['AMT_MAX_OVERDUE_RATIO'] = buro['AMT_CREDIT_MAX_OVERDUE']/(1+buro['AMT_CREDIT_SUM'])
buro['DAYS_END_DIFF'] = buro['DAYS_ENDDATE_FACT'] - buro['DAYS_CREDIT_ENDDATE']

#Label Encoding
categorical_feats = [
    f for f in buro.columns if buro[f].dtype == 'object'
]

for f_ in categorical_feats:
    nunique = buro[f_].nunique(dropna=False)
    print(f_,nunique,buro[f_].unique())
    buro[f_], indexer = pd.factorize(buro[f_])
    


# Aggragate the balance info for each buro record. Features include:
# * month account closed relative to current application
# * month with days past due (DPD) relative to current application
# * mean/sum/max DPD of each bureau account
#
# ...

bubl = pd.read_csv(f'{input_path}/bureau_balance.csv')
#what is the last month with DPD
bubl_last_DPD = bubl[bubl.STATUS.isin(['1','2','3','4','5'])].groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].max()
bubl_last_DPD.rename('MONTH_LAST_DPD', inplace=True)

#what is the last month complete
bubl_last_C = bubl[bubl.STATUS=='C'].groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].min()
bubl_last_C.rename('MONTH_LAST_C',inplace=True)

bubl['STATUS_DPD'] = bubl['STATUS']
bubl['STATUS_DPD'].loc[bubl['STATUS_DPD'].isin(['C','X'])]=np.nan
bubl['STATUS_DPD'] = bubl['STATUS_DPD'].astype('float')
bubl['YEAR_SCALE'] = (bubl['MONTHS_BALANCE']/12.0).apply(np.exp)
bubl['STATUS_DPD_SCALE'] = bubl['STATUS_DPD'] * bubl['YEAR_SCALE']
num_aggregations = {
    'STATUS_DPD': [ 'max', 'mean', 'sum'],
    'STATUS_DPD_SCALE': [ 'sum',],
    'YEAR_SCALE': [ 'sum']
}
balance = bubl.groupby('SK_ID_BUREAU').agg(num_aggregations)
balance.columns = pd.Index(['balance_' + e[0] + "_" + e[1].upper() for e in balance.columns.tolist()])
balance['balance_STATUS_DPD_SCALE_MEAN'] = balance['balance_STATUS_DPD_SCALE_SUM']/balance['balance_YEAR_SCALE_SUM']
del balance['balance_YEAR_SCALE_SUM']
gc.collect()
bubl_STATUS = pd.concat([bubl[['SK_ID_BUREAU','MONTHS_BALANCE']], pd.get_dummies(bubl['STATUS'], prefix='STATUS')], axis=1)
bubl_STATUS['STATUS_DPD'] = bubl_STATUS['STATUS_1'] + bubl_STATUS['STATUS_2'] + bubl_STATUS['STATUS_3'] + bubl_STATUS['STATUS_4'] + bubl_STATUS['STATUS_5'] 
num_aggregations = {
    'STATUS_C': [ 'sum'],
    'STATUS_X': [ 'sum'],
    'STATUS_0': [ 'sum'],
    'STATUS_DPD': ['sum']
}
balance_tot =  bubl_STATUS.groupby('SK_ID_BUREAU').agg(num_aggregations)
balance_12 =  bubl_STATUS.loc[bubl_STATUS['MONTHS_BALANCE']>=-12].groupby('SK_ID_BUREAU').agg(num_aggregations)
balance_tot.columns = pd.Index(['balance_tot_' + e[0] + "_" + e[1].upper() for e in balance_tot.columns.tolist()])
balance_12.columns = pd.Index(['balance_12_' + e[0] + "_" + e[1].upper() for e in balance_12.columns.tolist()])
balance_tot['balance_tot_STATUS_DPD_RATIO'] = balance_tot['balance_tot_STATUS_DPD_SUM']/(0.001 + balance_tot['balance_tot_STATUS_0_SUM'] + balance_tot['balance_tot_STATUS_DPD_SUM'])
balance_12['balance_12_STATUS_DPD_RATIO'] = balance_12['balance_12_STATUS_DPD_SUM']/(0.001 + balance_12['balance_12_STATUS_0_SUM'] + balance_12['balance_12_STATUS_DPD_SUM'])
balance = balance.merge(balance_tot, how='outer', on='SK_ID_BUREAU')             
balance = balance.merge(balance_12, how='outer', on='SK_ID_BUREAU')
balance['balance_has_DPD'] = (balance['balance_STATUS_DPD_MAX']>0).astype('int')

del balance_tot, balance_12, bubl_STATUS
gc.collect()


# Merge bureau balance feature with main bureau table:

buro_meta = buro.merge(balance, on='SK_ID_BUREAU', how='left')
del buro, balance
gc.collect()
print("bureau data shape", buro_meta.shape)


# Broadcast current target to bureau record, according to the current ID each bureau record correspond to.


target_map = pd.Series(data.TARGET.values, index=data.SK_ID_CURR.values)
y = buro_meta['SK_ID_CURR'].map(target_map)


# Split train and test set (test set are those without target)

train_x = buro_meta.loc[~y.isnull()]
test_x = buro_meta.loc[y.isnull()]
train_y = y.loc[~y.isnull()]


excluded_feats = ['SK_ID_CURR','SK_ID_BUREAU']
features = [f_ for f_ in train_x.columns.values if not f_ in excluded_feats]
print(excluded_feats)

train_x = buro_meta.loc[~y.isnull()]
test_x = buro_meta.loc[y.isnull()]
train_y = y.loc[~y.isnull()]

# Run a 5 fold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
oof_preds = np.zeros(train_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])
feature_importance_df = pd.DataFrame()


# Train LightGBM classifier


scores = []

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    trn_x, val_x = train_x[features].iloc[trn_idx], train_x[features].iloc[val_idx]
    trn_y, val_y = train_y.iloc[trn_idx], train_y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        metric = 'auc',
        num_leaves=20,
        colsample_bytree=0.8,
        subsample=0.9,
        max_depth=5,
        reg_alpha=5,
        reg_lambda=4,
        min_split_gain=0.002,
        min_child_weight=40,
        silent=True,
        verbose=-1,
        n_jobs = 16,
        random_state = n_fold * 619,
        scale_pos_weight = 2
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=60,
            categorical_feature = categorical_feats,
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
    sub_preds += clf.predict_proba(test_x[features])[:, 1] / folds.n_splits
    
    fold_score = roc_auc_score(val_y, oof_preds[val_idx])
    scores.append(fold_score)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, fold_score))
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f +- %0.4f' % (roc_auc_score(train_y, oof_preds), np.std(scores)))


# Get prediction for each bureau record -- giving each bureau record a score, which meatures how likely it belongs to a user who has defaulting account currently.


train_buro_score = train_x[['SK_ID_CURR','SK_ID_BUREAU','DAYS_CREDIT']]
train_buro_score['score'] = oof_preds
test_buro_score = test_x[['SK_ID_CURR','SK_ID_BUREAU','DAYS_CREDIT']]
test_buro_score['score'] = sub_preds
buro_score = pd.concat([train_buro_score,test_buro_score])
buro_score.to_csv(f'{current_path}/../../output/buro_score.csv',index=False,compression='zip')


# Group by current ID, create aggragated bureau score. These will be the features we use for final training.
# 
# Aggragated features include: mean, max, sum, variance, sum of past two year.
# 
# Note we subtract the global mean of all predictions, this is to prevent the "sum" feature penalized users with more accounts. The max/mean/var features are not affected by the substraction.


buro_score['score'] -= buro_score['score'].mean()
agg_buro_score = buro_score.groupby('SK_ID_CURR')['score'].agg({'max','mean','sum','var'})

agg_buro_score_recent2y = buro_score.loc[buro_score['DAYS_CREDIT']>-365.25*2].groupby('SK_ID_CURR')['score'].sum()

idx = buro_score.groupby(['SK_ID_CURR'])['DAYS_CREDIT'].idxmax()
agg_buro_score_last = buro_score[['SK_ID_CURR','score']].loc[idx.values]
agg_buro_score_last.set_index('SK_ID_CURR',inplace=True)

agg_buro_score['recent2y_sum'] = agg_buro_score_recent2y
agg_buro_score['last'] = agg_buro_score_last
agg_buro_score = agg_buro_score.add_prefix('buro_score_')
agg_buro_score['TARGET'] = target_map
agg_buro_score.to_csv(f'{current_path}/../../output/agg_buro_score.csv',compression='zip')
agg_buro_score.groupby('TARGET').mean()


# Check how the aggregated features are correlated to current target. Idealy we should see a significant correlation.

for col in agg_buro_score.columns:
    print(col,agg_buro_score[col].corr(agg_buro_score['TARGET']))


# Plot feature importance


# Plot feature importances
feature_importance = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)

best_features = feature_importance.iloc[:50].reset_index()

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(8, 16))
gs = gridspec.GridSpec(1, 1)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='importance', y='feature', data=best_features, ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)

