#!/usr/bin/env python
# coding: utf-8

# House and document features from logistic regression
# 仅使用房屋或文档特征，通过逻辑回归建模预测目标。
# 将预测值保存到磁盘中，稍后将与主表合并。


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import os

gc.enable()
current_path = os.path.dirname(__file__)
input_path = f'{current_path}/../../input'

# ### House features
# 
# Read house features from main train/test table.

data = pd.read_csv(f'{input_path}/application_train.csv')
test = pd.read_csv(f'{input_path}/application_test.csv')

rejected_features = ['AMT_GOODS_PRICE',
                     'APARTMENTS_AVG','APARTMENTS_MEDI',
                     'BASEMENTAREA_AVG','BASEMENTAREA_MODE','COMMONAREA_AVG','COMMONAREA_MODE',
                     'ELEVATORS_AVG','ELEVATORS_MEDI','ENTRANCES_AVG','ENTRANCES_MEDI','FLOORSMAX_AVG','FLOORSMAX_MEDI',
                     'FLOORSMIN_AVG','FLOORSMIN_MEDI','LANDAREA_AVG','LANDAREA_MODE',
                     'LIVINGAPARTMENTS_AVG','LIVINGAPARTMENTS_MEDI',
                     'LIVINGAREA_AVG','LIVINGAREA_MODE',
                     'NONLIVINGAPARTMENTS_AVG','NONLIVINGAPARTMENTS_MEDI',
                     'NONLIVINGAREA_AVG','NONLIVINGAREA_MODE','OBS_60_CNT_SOCIAL_CIRCLE',
                     'REGION_RATING_CLIENT_W_CITY','YEARS_BEGINEXPLUATATION_AVG','YEARS_BEGINEXPLUATATION_MEDI',
                     'YEARS_BUILD_AVG','YEARS_BUILD_MEDI']

for f_ in rejected_features:
    del data[f_]
    del test[f_]
    
gc.collect()


# Create per person house features (living area per person, number of floors per person, etc.)

data['AGE'] = - (data['DAYS_BIRTH']/365.25).astype('int32')
data['house_person'] = 1
data['house_person'].loc[data['NAME_HOUSING_TYPE']=='With parents'] +=2
data['house_person'].loc[(data['NAME_FAMILY_STATUS']=='Married')|(data['NAME_FAMILY_STATUS']=='Civil marriage')] +=1
data['house_person'].loc[data['AGE']<55] += data['CNT_CHILDREN']

test['AGE'] = - (test['DAYS_BIRTH']/365.25).astype('int32')
test['house_person'] = 1
test['house_person'].loc[test['NAME_HOUSING_TYPE']=='With parents'] +=2
test['house_person'].loc[(test['NAME_FAMILY_STATUS']=='Married')|(test['NAME_FAMILY_STATUS']=='Civil marriage')] +=1
test['house_person'].loc[test['AGE']<55] += test['CNT_CHILDREN']

house = [f_ for f_ in data.columns if ('AVG' in f_) | ('MEDI' in f_) | ('MODE' in f_) & (not 'YEARS' in f_)]
for f_ in house:
    if data[f_].dtype != 'object':
        print (f_)
        data[f_+'_PP'] = data[f_]/data['house_person']
        test[f_+'_PP'] = test[f_]/test['house_person']


# In[4]:


house = [f_ for f_ in data.columns if ('AVG' in f_) | ('MEDI' in f_) | ('MODE' in f_) ]
print (len(house),house)
train_house = data.loc[data[house].isnull().sum(axis=1)<len(house)][['REGION_POPULATION_RELATIVE','NAME_HOUSING_TYPE']+house]
test_house = test.loc[test[house].isnull().sum(axis=1)<len(house)][['REGION_POPULATION_RELATIVE','NAME_HOUSING_TYPE']+house]
y = data['TARGET'].loc[data[house].isnull().sum(axis=1)<len(house)]
train_ID = data['SK_ID_CURR'].loc[data[house].isnull().sum(axis=1)<len(house)]
test_ID = test['SK_ID_CURR'].loc[test[house].isnull().sum(axis=1)<len(house)]


# Create one hot encoding for some features.

train_house_size = train_house.shape[0]
test_house_size = test_house.shape[0]
combined = pd.concat([train_house, test_house], axis=0)
for f_ in combined.columns:
    if combined[f_].dtype == 'object':
        combined[f_].fillna(combined[f_].mode(), inplace=True)
    else:
        combined[f_].fillna(combined[f_].median(), inplace=True)
combined = pd.concat([combined, pd.get_dummies(combined['REGION_POPULATION_RELATIVE'], prefix='REGION')], axis=1)
for f_ in ['FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','NAME_HOUSING_TYPE']:
    combined = pd.concat([combined, pd.get_dummies(combined[f_], prefix=f_)], axis=1)
    del combined[f_]
train_house = combined.iloc[:train_house_size,:]
test_house = combined.iloc[-test_house_size:,:]


# #### Training using current target.


from sklearn.linear_model import LogisticRegression

train_x = train_house
train_y = y
test_x = test_house
print(train_x.shape, test_x.shape)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(train_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])
feature_importance_df = pd.DataFrame()

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    trn_x, val_x = train_x.iloc[trn_idx], train_x.iloc[val_idx]
    trn_y, val_y = train_y.iloc[trn_idx], train_y.iloc[val_idx]
     
    model = LogisticRegression(C=0.05)
    model.fit(trn_x, trn_y)
    
    trn_y_pred = model.predict_proba(trn_x)[:,1]
    oof_preds[val_idx] = model.predict_proba(val_x)[:,1]
    sub_preds += model.predict_proba(test_x)[:,1] / folds.n_splits
    print('Fold %2d AUC : (train): %.6f, (val): %.6f' % (n_fold + 1, 
        roc_auc_score(trn_y, trn_y_pred), roc_auc_score(val_y, oof_preds[val_idx])))
    
print('Full AUC score %.6f' % roc_auc_score(train_y, oof_preds)) 


# Save predictions to disk.

train_house_score = pd.DataFrame({'house_score':oof_preds}, index=train_ID)
test_house_score = pd.DataFrame({'house_score':sub_preds}, index=test_ID)
train_house_score.to_csv(f'{current_path}/../../output/train_house_score.csv')
test_house_score.to_csv(f'{current_path}/../../output/test_house_score.csv')

del combined, trn_x, trn_y, train_x, train_y, test_x
del train_house, test_house
gc.collect()


# #### Another training ues credit score as target


house = [f_ for f_ in data.columns if ('AVG' in f_) | ('MEDI' in f_) | ('MODE' in f_) ]
combined = pd.concat([data,test],axis=0,sort=False)
combined_house = combined.loc[combined[house].isnull().sum(axis=1)<47][['REGION_POPULATION_RELATIVE','NAME_HOUSING_TYPE']+house]
y = combined[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].loc[combined[house].isnull().sum(axis=1)<47].mean(axis=1)
combined_ID = combined['SK_ID_CURR'].loc[combined[house].isnull().sum(axis=1)<47]

combined_house = combined_house.loc[y.notna()]
combined_ID = combined_ID.loc[y.notna()]
y = y.loc[y.notna()]
print (y.shape, combined_house.shape, combined_ID.shape)


combined = combined_house
for f_ in combined.columns:
    if combined[f_].dtype == 'object':
        combined[f_].fillna(combined[f_].mode(), inplace=True)
    else:
        combined[f_].fillna(combined[f_].median(), inplace=True)
combined = pd.concat([combined, pd.get_dummies(combined['REGION_POPULATION_RELATIVE'], prefix='REGION')], axis=1)
for f_ in ['FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','NAME_HOUSING_TYPE']:
    combined = pd.concat([combined, pd.get_dummies(combined[f_], prefix=f_)], axis=1)
    del combined[f_]


# This time we use Ridge because credit scores are continuous...

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

folds = KFold(n_splits=5, shuffle=True, random_state=546789)

train_x = combined
train_y = y

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
    trn_x, val_x = train_x.iloc[trn_idx], train_x.iloc[val_idx]
    trn_y, val_y = train_y.iloc[trn_idx], train_y.iloc[val_idx]
     
    model = Ridge()    
    model.fit(trn_x, trn_y)
    
    trn_y_pred = model.predict(trn_x)
    val_y_pred = model.predict(val_x)
    print('Fold %2d r2_score : (train): %.6f, (val): %.6f' % (n_fold + 1, 
        r2_score(trn_y, trn_y_pred), r2_score(val_y, val_y_pred)))

model.fit(train_x, y)
pred_y = model.predict(train_x)
print('Full r2_score %.6f' % r2_score(train_y, pred_y)) 


# Save predictions to disk.

house_ex = pd.DataFrame({'house_score':pred_y}, index=combined_ID)
house_ex.to_csv(f'{current_path}/../../output/house_ex.csv')

del combined, combined_house, train_x, train_y
gc.collect()


# ### Document features
# 
# Read document features from main train/test table.


doc = [f_ for f_ in data.columns if ('FLAG_DOCUMENT' in f_) ]
train_x = data[doc]
test_x = test[doc]
train_y = data['TARGET']
train_ID = data['SK_ID_CURR']
test_ID = test['SK_ID_CURR']


# Training with logistic regression.


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(train_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    trn_x, val_x = train_x.iloc[trn_idx], train_x.iloc[val_idx]
    trn_y, val_y = train_y.iloc[trn_idx], train_y.iloc[val_idx]
     
    model = LogisticRegression(C=0.1)
    model.fit(trn_x, trn_y)
    
    trn_y_pred = model.predict_proba(trn_x)[:,1]
    oof_preds[val_idx] = model.predict_proba(val_x)[:,1]
    sub_preds += model.predict_proba(test_x)[:,1] / folds.n_splits
    print('Fold %2d AUC : (train): %.6f, (val): %.6f' % (n_fold + 1, 
        roc_auc_score(trn_y, trn_y_pred), roc_auc_score(val_y, oof_preds[val_idx])))
    
print('Full AUC score %.6f' % roc_auc_score(train_y, oof_preds)) 


# Save predictions to disk.


train_doc_score = pd.DataFrame({'doc_score':oof_preds}, index=train_ID)
test_doc_score = pd.DataFrame({'doc_score':sub_preds}, index=test_ID)
train_doc_score.to_csv(f'{current_path}/../../output/train_doc_score.csv')
test_doc_score.to_csv(f'{current_path}/../../output/test_doc_score.csv')

del trn_x, trn_y, train_x, train_y, test_x
gc.collect()

