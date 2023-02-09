#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score

train_sub1 = pd.read_csv('../output/train_pred_lgb1.csv')
train_sub2 = pd.read_csv('../output/train_pred_lgb2.csv')
train_sub3 = pd.read_csv('../output/train_pred_lgb3.csv')

test_sub1 = pd.read_csv('../output/test_pred_lgb2_copy2.csv')
test_sub2 = pd.read_csv('../output/test_pred_lgb2.csv')
test_sub3 = pd.read_csv('../output/test_pred_lgb2_copy.csv')

train = train_sub1.copy()
train.rename({'prob': 'sub1'}, axis=1, inplace=True)
train['sub2'] = train_sub2['prob']
train['sub3'] = train_sub3['prob']

train = train[['SK_ID_CURR', 'sub1', 'sub2', 'sub3', 'target']]

test = test_sub1.copy()
test.rename({'TARGET': 'sub1'}, axis=1, inplace=True)
test['sub2'] = test_sub2['TARGET']
test['sub3'] = test_sub3['TARGET']


all_preds = train[['sub1', 'sub2', 'sub3']].values

def max_auc(params):
    preds = None
    for index, val in enumerate(params.keys()):
        if index == 0:
            preds = params[val]*all_preds[:, 0]
        else:
            preds += params[val]*all_preds[:, index]
    
    param_sum = 0
    for key, val in params.items():
        param_sum += val

    preds = preds/param_sum
    score = roc_auc_score(train['target'], preds)
    return score

def objective(trial):
    params = {}
    for i in range(3):
        params[f"w{i+1}"] = trial.suggest_float(f'w{i+1}', 0, 1)
    score = max_auc(params)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

print(study.best_params)

weights = list(study.best_params.values())
weights = [w / sum(weights) for w in weights]

final_pred = None
for i, model in enumerate(['sub1', 'sub2', 'sub3']):
    if i == 0:
        final_pred = test[model] * weights[i]
    else:
        final_pred += test[model] * weights[i]

sub = test[['SK_ID_CURR']]
sub['TARGET'] = final_pred
sub.to_csv('../output/ensemble.csv', index = False)
