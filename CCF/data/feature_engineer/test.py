# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-9-11 下午7:06
================================='''
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, r2_score
from hyperopt import fmin, tpe, hp, partial
from numpy.random import random, shuffle
import matplotlib.pyplot as plt
from pandas import DataFrame
import tensorflow as tf
# from PIL import Image
import lightgbm as lgb
import networkx as nx
import pandas as pd
import numpy as np
import warnings
# import cv2
import os
import re

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
context_data=pd.read_csv('data/all_train_context_data.csv')
context_data['date']=((context_data.regYear-2016)*12+context_data.regMonth).tolist()

del context_data['regMonth'],context_data['regYear'],context_data['province'],context_data['label']


context_data['price']=context_data['all_body_except']*context_data['province_all_body_attention']
print(context_data.columns.tolist())
# target_aim=['price']
# target_aim=['all_body_except']
target_aim=['province_all_body_attention']

origin_cols=['date','adcode', 'model', 'bodyType','popularity','carCommentVolum', 'newsReplyVolum']+target_aim

itmer=20
for i in itmer:
    print()
    print()

train_data=context_data[context_data['date']<21][origin_cols]
vail_data=context_data[context_data['date']>=21][origin_cols]
vail_data[target_aim]=vail_data[target_aim]*10000

cate_col=['date','bodyType','adcode','model']
for col in cate_col:
    train_data[col]=train_data[col].astype('category')
    vail_data[col]=vail_data[col].astype('category')

# print(train_data[target_aim].head())


print(train_data.info())

lgb_model = lgb.LGBMRegressor(
    num_leaves=32, reg_alpha=1, reg_lambda=0.1, objective='mse',
    max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=4329,
    n_estimators=5000, subsample=0.8, colsample_bytree=0.8,
)

lgb_model.fit(
    train_data[origin_cols],train_data[target_aim],
    eval_set=[(vail_data[origin_cols],vail_data[target_aim])],
    categorical_feature=cate_col, early_stopping_rounds=100,
    verbose=100,
)


lgb_model.n_estimators=900
lgb_model.predict()
print(lgb_model.best_score_)