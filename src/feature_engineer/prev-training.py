#!/usr/bin/env python
# coding: utf-8

# Previous application training

# 这个代码尝试利用历史申请与当前违约率之间的相关性。
# 历史的一次申请作为一条训练样本，标签是对应的当前标签: 属于同一客户的历史申请都将使用该客户的当前标签。
# 使用LightGBM分类器来预测每次历史申请属于当前有违约贷款的客户的概率。
# 模型训练完成后，按照当前客户ID聚合分组，对预测结果进行进行统计，计算出均值/总和等统计信息。
# 将这些统计信息保存到磁盘中，以便与主表合并。


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


# Create features for each previous application. Apart from raw features from the previous application table, we also compute:
# * ratio between credit applied in debt and credit granted
# * ratio between credit granted and annuity -- expected time to payoff the loan
# * amount left to pay at the time of current application.
# * difference between actual and expected last payment date
# 
# ...


prev = pd.read_csv(f'{input_path}/previous_application.csv')
prev = prev.loc[prev['FLAG_LAST_APPL_PER_CONTRACT']=='Y'] #mistake rows
del prev['FLAG_LAST_APPL_PER_CONTRACT']

#replace strange number of days as nan
for f_ in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
    prev[f_].loc[prev[f_]>360000] = np.nan

#create some features
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
prev['AMT_DIFF_CREAPP'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
prev['AMT_DIFF_CREDIT_GOODS'] = prev['AMT_CREDIT'] - prev['AMT_GOODS_PRICE']
prev['AMT_CREDIT_GOODS_PERC'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']
prev['AMT_PAY_YEAR'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
prev['DAYS_TOTAL'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']
prev['DAYS_TOTAL2'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']
prev['AMT_LEFT'] = (prev['AMT_CREDIT'] - prev['AMT_ANNUITY'] * prev['DAYS_LAST_DUE_1ST_VERSION']/365.25).clip(lower=0)
prev['PAYMENT_LEFT'] = prev['AMT_LEFT']/prev['AMT_ANNUITY']

#these features highly correlated with others or not useful?
rejected_features = ['AMT_GOODS_PRICE',
                     'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                     'NFLAG_LAST_APPL_IN_DAY']
for f_ in rejected_features:
    del prev[f_]
    
#Label Encoding
categorical_feats = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]

for f_ in categorical_feats:
    nunique = prev[f_].nunique(dropna=False)
    print(f_,nunique,prev[f_].unique())
    prev[f_], indexer = pd.factorize(prev[f_])
    


# Aggragate the installment features for each buro record. Some new features are:
# * difference between expected and actural payment amount.
# * difference between expected and actural payment date.
# * last time DPD.
# 
# ...
# 
# Compute stats (mean/max/sum...) of these new and raw features over time.

inst = pd.read_csv(f'{input_path}/installments_payments.csv')
inst_NUM_INSTALMENT_VERSION = inst.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique()

#merge payments of same month
#maybe helpful for: inst.loc[(inst.SK_ID_PREV==1000005) & (inst.SK_ID_CURR==176456) & (inst.NUM_INSTALMENT_NUMBER==9)]
inst['DAYS_ENTRY_PAYMENT_weighted'] = inst['DAYS_ENTRY_PAYMENT'] * inst['AMT_PAYMENT']
inst['MONTHS_BALANCE'] = (inst['DAYS_INSTALMENT']/30.4375).astype('int')
inst = inst.groupby(['SK_ID_PREV','SK_ID_CURR','MONTHS_BALANCE']).agg({'DAYS_INSTALMENT':'mean',
                                                                       'DAYS_ENTRY_PAYMENT_weighted':'sum',
                                                                       'AMT_INSTALMENT':'mean',
                                                                       'AMT_PAYMENT':'sum'})
inst['DAYS_ENTRY_PAYMENT'] = inst['DAYS_ENTRY_PAYMENT_weighted']/inst['AMT_PAYMENT']
inst = inst.reset_index()
del inst['DAYS_ENTRY_PAYMENT_weighted']

inst['AMT_PAYMENT_PERC'] = inst['AMT_PAYMENT'] / (1+inst['AMT_INSTALMENT'])
inst['AMT_PAYMENT_DIFF'] = inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']
inst['DPD'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
inst['DBD'] = (inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']).clip(lower=0)
inst['DPD'].fillna(30,inplace=True)
inst['DBD'].fillna(0,inplace=True)

#when is the last time late
inst_last_late = inst[inst.DAYS_INSTALMENT < inst.DAYS_ENTRY_PAYMENT].groupby(['SK_ID_PREV'])['DAYS_INSTALMENT'].max()
inst_last_late.rename('DAYS_LAST_LATE',inplace=True)

#when is the last time underpaid
inst_last_underpaid = inst[inst.AMT_INSTALMENT < inst.AMT_PAYMENT].groupby(['SK_ID_PREV'])['DAYS_INSTALMENT'].max()
inst_last_underpaid.rename('DAYS_LAST_UNDERPAID',inplace=True)

num_aggregations = {
    'MONTHS_BALANCE': ['size','min','max'],
    'AMT_PAYMENT_PERC': ['max','mean','var'],
    'AMT_PAYMENT_DIFF': [ 'sum', 'mean','var'],
    'AMT_PAYMENT': [ 'sum','mean','var'],
    'DPD': ['sum', 'max','mean','var'],
    'DBD': ['sum', 'max','mean','var'],
}
inst = inst.groupby('SK_ID_PREV').agg(num_aggregations)
inst.columns = pd.Index([e[0] + "_" + e[1].upper() for e in inst.columns.tolist()])
inst['N_NUM_INSTALMENT_VERSION'] = inst_NUM_INSTALMENT_VERSION
inst['DAYS_LAST_LATE'] = inst_last_late
inst['DAYS_LAST_UNDERPAID'] = inst_last_underpaid
inst = inst.add_prefix('inst_')


# Aggragate the pos cash features for each buro record.

pos = pd.read_csv(f'{input_path}/POS_CASH_balance.csv')
idx = pos.groupby(['SK_ID_PREV'])['MONTHS_BALANCE'].idxmax() #most recent data
pos_recent = pos[['SK_ID_PREV','MONTHS_BALANCE','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                  'NAME_CONTRACT_STATUS','SK_DPD','SK_DPD_DEF']].loc[idx.values]
pos_recent['NAME_CONTRACT_STATUS'],indexer = pd.factorize(pos_recent['NAME_CONTRACT_STATUS'])
pos_recent.set_index('SK_ID_PREV',inplace=True)
pos_recent.columns = ['pos_recent_' + f_ for f_ in pos_recent.columns]

#what is the last month with DPD
pos_last_DPD = pos[pos.SK_DPD>0].groupby(['SK_ID_PREV'])['MONTHS_BALANCE'].max()
pos_last_DPD.rename('MONTH_LAST_DPD',inplace=True)

pos['has_DPD'] = 0
pos['has_DPD'].loc[pos['SK_DPD']>0] = 1
num_aggregations = {
    'MONTHS_BALANCE': ['size'],
    'has_DPD': ['sum','mean'],
    'SK_DPD': ['max','mean'],
    'SK_DPD_DEF': [ 'sum', 'median'],
}
pos = pos.groupby('SK_ID_PREV').agg(num_aggregations)
pos.columns = pd.Index(['pos_' + e[0] + "_" + e[1].upper() for e in pos.columns.tolist()])
pos = pos.merge(pos_recent, how='outer', on='SK_ID_PREV')

pos['MONTH_LAST_DPD'] = pos_last_DPD
del pos_recent
gc.collect


# Aggragate the credit card features for each buro record.


ccbl = pd.read_csv(f'{input_path}/credit_card_balance.csv')
ccbl['AMT_BALANCE_CREDIT_RATIO'] = (ccbl['AMT_BALANCE']/(ccbl['AMT_CREDIT_LIMIT_ACTUAL']+0.001)).clip(-100,100)
ccbl['AMT_CREDIT_USE_RATIO'] = (ccbl['AMT_DRAWINGS_CURRENT']/(ccbl['AMT_CREDIT_LIMIT_ACTUAL']+0.001)).clip(-100,100)
ccbl['AMT_DRAWING_ATM_RATIO'] = ccbl['AMT_DRAWINGS_ATM_CURRENT']/(ccbl['AMT_DRAWINGS_CURRENT']+0.001)
ccbl['AMT_PAY_USE_RATIO'] = ((ccbl['AMT_PAYMENT_TOTAL_CURRENT']+0.001)/(ccbl['AMT_DRAWINGS_CURRENT']+0.001)).clip(-100,100)
ccbl['AMT_BALANCE_RECIVABLE_RATIO'] = ccbl['AMT_BALANCE']/(ccbl['AMT_TOTAL_RECEIVABLE']+0.001)
ccbl['AMT_DRAWING_BALANCE_RATIO'] = ccbl['AMT_DRAWINGS_CURRENT']/(ccbl['AMT_BALANCE']+0.001)
ccbl['AMT_RECEIVABLE_PRINCIPAL_DIFF'] = ccbl['AMT_TOTAL_RECEIVABLE']-ccbl['AMT_RECEIVABLE_PRINCIPAL']
ccbl['AMT_PAY_INST_DIFF'] = ccbl['AMT_PAYMENT_CURRENT'] - ccbl['AMT_INST_MIN_REGULARITY']

rejected_features = ['AMT_RECIVABLE','AMT_RECEIVABLE_PRINCIPAL',
                     'AMT_DRAWINGS_POS_CURRENT']
for f_ in rejected_features:
    del ccbl[f_]

ccbl_last_DPD = ccbl[ccbl.SK_DPD>0].groupby(['SK_ID_CURR'])['MONTHS_BALANCE'].max()
ccbl_last_DPD.rename('MONTH_LAST_DPD',inplace=True)

sum_feats = [f_ for f_ in ccbl.columns.values[3:] if (f_.find('RATIO')==-1) & (f_.find('CNT')==-1)]
print ('sum_feats', sum_feats)
ccbl_sum =  ccbl.groupby('SK_ID_PREV')[sum_feats].sum()
ccbl_sum = ccbl_sum.add_prefix('sum_')

mean_feats = [f_ for f_ in ccbl.columns.values[3:]]
print ('mean_feats', mean_feats)
ccbl_mean = ccbl.groupby('SK_ID_PREV')[mean_feats].mean()
ccbl_mean = ccbl_mean.add_prefix('mean_')

ccbl = ccbl_mean.merge(ccbl_sum, how='outer', on='SK_ID_PREV')

ccbl['last_DPD'] = ccbl_last_DPD
ccbl = ccbl.add_prefix('cc_')
del ccbl_mean, ccbl_sum
gc.collect()


# Merge installment, pos cash, credit card features to the main previous application table:


prev_meta = prev.merge(inst, on='SK_ID_PREV', how='left')
prev_meta = prev_meta.merge(pos, on='SK_ID_PREV', how='left')
prev_meta = prev_meta.merge(ccbl, on='SK_ID_PREV', how='left')
del inst, pos, ccbl
gc.collect()


# Create a few more features:

interest_tmp = prev_meta['RATE_INTEREST_PRIVILEGED'].fillna(0)
downpayment_tmp = prev_meta['AMT_DOWN_PAYMENT'].fillna(0)
inst_AMT_PAYMENT_SUM_tmp = prev_meta['inst_AMT_PAYMENT_SUM'].fillna(0)
prev_meta['AMT_LEFT2'] = (prev_meta['AMT_CREDIT']-downpayment_tmp)*(1+interest_tmp) - inst_AMT_PAYMENT_SUM_tmp
prev_meta['AMT_LEFT2'] = prev_meta['AMT_LEFT2'].clip(lower=0)
prev_meta['AMT_LEFT2'].loc[prev_meta['NAME_CONTRACT_STATUS']!=0] = 0
prev_meta['PAYMENT_CREDIT_RATIO'] = prev_meta['inst_AMT_PAYMENT_SUM']/prev_meta['AMT_CREDIT']
prev_meta['AMT_LEFT3'] = prev_meta['AMT_CREDIT'] * prev_meta['pos_recent_CNT_INSTALMENT']/ (prev_meta['pos_recent_CNT_INSTALMENT_FUTURE'] + prev_meta['pos_recent_CNT_INSTALMENT'])


# Broadcast current target to previous applications, according to the current ID each previous application correspond to.


target_map = pd.Series(data.TARGET.values, index=data.SK_ID_CURR.values)
y = prev_meta['SK_ID_CURR'].map(target_map)


# Split train and test set (test set are those without target)

train_x = prev_meta.loc[~y.isnull()]
test_x = prev_meta.loc[y.isnull()]
train_y = y.loc[~y.isnull()]


excluded_feats = ['SK_ID_CURR','SK_ID_PREV']
features = [f_ for f_ in train_x.columns.values if not f_ in excluded_feats]
print(excluded_feats)

train_x = prev_meta.loc[~y.isnull()]
test_x = prev_meta.loc[y.isnull()]
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
        num_leaves=20,
        metric = 'auc',
        colsample_bytree=0.3,
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
            eval_metric='auc', verbose=100, early_stopping_rounds=50,
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


# Get prediction for each previous application -- giving each previous application a score, which meatures how likely it belongs to a user who has defaulting account currently.

train_prev_score = train_x[['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION']]
train_prev_score['score'] = oof_preds
test_prev_score = test_x[['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION']]
test_prev_score['score'] = sub_preds
prev_score = pd.concat([train_prev_score,test_prev_score])
prev_score.to_csv(f'{current_path}/../../output/prev_score.csv',index=False,compression='zip')


# Group by current ID, create aggragated previous score. These will be the features we use for final training.
# 
# Aggragated features include: mean, max, sum, variance, sum of past two year.
# 
# Note we subtract the global mean of all predictions, this is to prevent the "sum" feature penalized users with more accounts. The max/mean/var features are not affected by the substraction.


agg_prev_score = prev_score.groupby('SK_ID_CURR')['score'].agg({'max','mean','sum','var'})

agg_prev_score_recent2y = prev_score.loc[prev_score['DAYS_DECISION']>-365.25*2].groupby('SK_ID_CURR')['score'].sum()

idx = prev_score.groupby(['SK_ID_CURR'])['DAYS_DECISION'].idxmax()
agg_prev_score_last = prev_score[['SK_ID_CURR','score']].loc[idx.values]
agg_prev_score_last.set_index('SK_ID_CURR',inplace=True)

agg_prev_score['recent2y_sum'] = agg_prev_score_recent2y
agg_prev_score['last'] = agg_prev_score_last
agg_prev_score = agg_prev_score.add_prefix('prev_score_')
agg_prev_score['TARGET'] = target_map
agg_prev_score.to_csv(f'{current_path}/../../output/agg_prev_score.csv',compression='zip')
agg_prev_score.groupby('TARGET').mean()


# Check how the aggregated features are correlated to current target. Idealy we should see a significant correlation.


for col in agg_prev_score.columns:
    print(col,agg_prev_score[col].corr(agg_prev_score['TARGET']))


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

