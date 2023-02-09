#!/usr/bin/env python
# coding: utf-8

# 这个代码用于找出客户每月行为与当前违约率之间的相关性。
# 按月分组创建训练集。
# 标签是当前标签进行扩散：同一用户的所有月记录共享相同的标签。
# 使用LightGBM分类器来预测某个月记录属于当前违约用户的概率。
# 按照当前客户ID聚合分组，对预测结果进行进行统计，计算出均值/总和等统计信息，创建后续将合并到主表中的特征。

# 需要注意的是，由于同一贷款的月度记录可能共享某些特征的相同值：例如每月付款的相同金额。
# 这有可能会导致模型leak，为避免leak，在做交叉验证时将同一客户的记录放在同一折中，使得模型无法利用这类信息。


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from lightgbm import LGBMClassifier, LGBMRegressor
import gc
import os

gc.enable()
current_path = os.path.dirname(__file__)
input_path = f'{current_path}/../../input'


# Aggragate pos cash.

pos = pd.read_csv(f'{input_path}/POS_CASH_balance.csv')
num_aggragations = {
    'SK_ID_PREV': ['count'],
    'CNT_INSTALMENT': ['mean','max','sum'],
    'CNT_INSTALMENT_FUTURE': ['mean','max','sum'],
    'SK_DPD': ['mean','max','sum'], 
    'SK_DPD_DEF': ['mean','max','sum'], 
}
pos = pos.groupby(['SK_ID_CURR','MONTHS_BALANCE']).agg(num_aggragations)
pos.columns = pd.Index(['pos_' + e[0] + "_" + e[1].upper() for e in pos.columns.tolist()])


# Aggragate installment payments.

inst = pd.read_csv(f'{input_path}/installments_payments.csv')
inst_NUM_INSTALMENT_VERSION = inst.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique()

#merge payments of same month
#maybe helpful for: inst.loc[(inst.SK_ID_PREV==1000005) & (inst.SK_ID_CURR==176456) & (inst.NUM_INSTALMENT_NUMBER==9)]
inst['DAYS_ENTRY_PAYMENT_weighted'] = inst['DAYS_ENTRY_PAYMENT'] * inst['AMT_PAYMENT']
inst['MONTHS_BALANCE'] = (inst['DAYS_INSTALMENT']/30.4375-1).astype('int')
inst = inst.loc[inst['MONTHS_BALANCE']>=-96]
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
inst['DPD'].fillna(inst['DPD'].max(),inplace=True)
inst['DBD'].fillna(0,inplace=True)

num_aggragations = {
    'SK_ID_PREV': ['count'],
    'AMT_PAYMENT_PERC': ['mean','max'],
    'AMT_PAYMENT_DIFF': ['mean','max','sum'],
    'DPD': ['mean','max','sum'], 
    'DBD': ['mean','max','sum'], 
}
inst = inst.groupby(['SK_ID_CURR','MONTHS_BALANCE']).agg(num_aggragations)
inst.columns = pd.Index(['inst_' + e[0] + "_" + e[1].upper() for e in inst.columns.tolist()])


# Aggragate buro balance.

buro = pd.read_csv(f'{input_path}/bureau.csv',usecols=['SK_ID_CURR','SK_ID_BUREAU'])
buro_map = pd.Series(buro['SK_ID_CURR'].values, index = buro['SK_ID_BUREAU'].values)

bubl = pd.read_csv(f'{input_path}/bureau_balance.csv')
bubl = bubl.loc[(bubl['STATUS']!='C')&(bubl['STATUS']!='X')]
bubl['SK_ID_CURR'] = bubl['SK_ID_BUREAU'].map(buro_map)
bubl = bubl[~bubl['SK_ID_CURR'].isnull()]
bubl['SK_ID_CURR'] = bubl['SK_ID_CURR'].astype('int')
bubl['STATUS'] = bubl['STATUS'].astype('int')

num_aggragation = {
    'SK_ID_BUREAU': ['count'],
    'STATUS': ['max','mean','sum']
}
bubl = bubl.groupby(['SK_ID_CURR','MONTHS_BALANCE']).agg(num_aggragation)
bubl.columns = pd.Index(['bubl_' + e[0] + "_" + e[1].upper() for e in bubl.columns.tolist()])


# Merge pos-cash, installment and buro balance. We have skipped credit card table because it consists of relatively few records. Include credit card table would results in very large training dataframe that do not have significant improvement.


alldata = pos.merge(inst, on=['SK_ID_CURR','MONTHS_BALANCE'], how='outer')
alldata = alldata.merge(bubl, on=['SK_ID_CURR','MONTHS_BALANCE'], how='outer')
alldata = alldata.reset_index()

print(alldata.shape)
del pos, bubl, inst, buro#, ccbl
gc.collect()

#downcasting to save space
for col in alldata.columns.values[2:]:
    alldata[col] = alldata[col].astype('float32')


# Map current target to our training table according the current ID each monthly record belongs to.

data = pd.read_csv(f'{input_path}/application_train.csv', usecols=['SK_ID_CURR','TARGET'])
target_map = pd.Series(data.TARGET.values, index=data.SK_ID_CURR.values)
y = alldata['SK_ID_CURR'].map(target_map)


# Split train and test.

train_x = alldata.loc[~y.isnull()]
test_x = alldata.loc[y.isnull()]
train_y = y.loc[~y.isnull()]

excluded_feats = ['SK_ID_CURR']
features = [f_ for f_ in train_x.columns.values if not f_ in excluded_feats]
print(excluded_feats)

# Run a 5 fold
oof_preds = np.zeros(train_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])
feature_importance_df = pd.DataFrame()


# Create KFold based on current ID. Monthly records belonging to the same current ID go to the same fold, this helps prevent the leak we have discussed.
# With this special CV the boosting stops at ~600 iterations, without it the boosting would easily reach +2000 rounds.


n_fold = 5
foldmap = pd.Series(np.random.randint(low=0, high=n_fold, size=alldata.SK_ID_CURR.nunique()), index=alldata.SK_ID_CURR.unique())
fold = train_x['SK_ID_CURR'].map(foldmap)
fold.value_counts()


# Train LGBMClassifier

scores = []

for i in range(0,n_fold):
    trn_idx = (fold != i)
    val_idx = (fold == i)
    trn_x, val_x = train_x[features].loc[trn_idx], train_x[features].loc[val_idx]
    trn_y, val_y = train_y.loc[trn_idx], train_y.loc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=16,
        metric = 'auc',
        colsample_bytree=0.3,
        subsample=0.5,
        subsample_freq = 1,
        max_depth=4,
        reg_alpha=5,
        reg_lambda=10,
        min_split_gain=0.004,
        min_child_weight=1000,
        silent=True,
        verbose=-1,
        n_jobs = 16,
        random_state = n_fold * 666,
        scale_pos_weight = 1
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=50,
            #categorical_feature = categorical_feats,
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
    sub_preds += clf.predict_proba(test_x[features])[:, 1] / n_fold
    
    fold_score = roc_auc_score(val_y, oof_preds[val_idx])
    scores.append(fold_score)
    print('Fold %2d AUC : %.6f' % (i + 1, fold_score))
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f +- %0.4f' % (roc_auc_score(train_y, oof_preds), np.std(scores)))


# Get the prediction as a score of each monthly record. The score measures how likely a certain monthly record belongs to someone who has defaulted loan currently. Substract the global mean to the score, this prevent penalizing customer with longer records.

train_month_score = train_x[['SK_ID_CURR','MONTHS_BALANCE']]
train_month_score['score'] = oof_preds
test_month_score = test_x[['SK_ID_CURR','MONTHS_BALANCE']]
test_month_score['score'] = sub_preds
month_score = pd.concat([train_month_score,test_month_score])
month_score['score_sub'] = month_score['score'] - month_score['score'].mean()
#month_score.to_csv(f'{current_path}/../../output/month_score.csv',index=False,compression='zip')


# Group by current ID and compute stats of month scores. The aggragated scores are saved to disk and be ready to use as features in the final training.

agg_month_score = month_score.groupby('SK_ID_CURR')['score_sub'].agg({'max','mean','std','sum'})
agg_month_score = agg_month_score.add_prefix("month_score_")
agg_month_score['TARGET'] = target_map
agg_month_score.to_csv(f'{current_path}/../../output/agg_month_score.csv',compression='zip')
agg_month_score.groupby('TARGET').mean()


