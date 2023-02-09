#!/bin/bash

# feature_engineer
nohup python ./src/feature_engineer/prev-training.py > ./log/prev-training.log 2>&1 &
nohup python ./src/feature_engineer/buro-training.py > ./log/buro-training.log 2>&1 &
nohup python ./src/feature_engineer/month-training.py > ./log/month-training.log 2>&1 &
nohup python ./src/feature_engineer/house-doc-feats.py > ./log/house-doc-feats.log 2>&1 &
nohup python ./src/feature_engineer/inst-ts.py > ./log/inst-ts.log 2>&1 &
nohup python ./src/feature_engineer/bubl-ts.py > ./log/bubl-ts.log 2>&1 &
nohup python ./src/feature_engineer/pos-ts.py > ./log/pos-ts.log 2>&1 &
nohup python ./src/feature_engineer/cc-ts.py > ./log/cc-ts.log 2>&1 &


# model
nohup python ./src/model/lgb1.py > ./log/lgb1.log 2>&1 &
nohup python ./src/model/lgb2.py > ./log/lgb2.log 2>&1 &
nohup python ./src/model/lgb3.py > ./log/lgb3.log 2>&1 &


# ensemble
nohup python ./src/ensemble/opt_weights.py > ./log/ensemble.log 2>&1 &
