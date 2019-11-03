#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/5 上午8:41
# @Author  : TIXhjq

from sklearn.model_selection import StratifiedKFold, KFold
from pandas import DataFrame
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import lightgbm as lgb
import numpy as np
import pandas as pd
import datetime
import time
import os
import gc

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

#-------------------------------------------------
folder='data/'

model_data=pd.read_csv(folder+'model_context.csv')
province_data=pd.read_csv(folder+'province_context.csv')
bodyType_data=pd.read_csv(folder+'bodyType_context.csv')

#origin
train_data=pd.read_csv(folder+'all_train_context_data.csv')
test_data=pd.read_csv(folder+'evaluation_public.csv')

bodyType_data_=train_data[['model','bodyType']].drop_duplicates('model')
test_data=test_data.merge(bodyType_data_,'left',on=['model'])

date_list=[]
for month,year in zip(test_data.regMonth.tolist(),test_data.regYear.tolist()):
    date=pd.to_datetime(str(year)+'-'+str(month))
    date_list.append(date)

test_data['regDate']=date_list

print(test_data.head())

feature_cols=train_data.columns.tolist()
group_list=['model','province','bodyType']
drop_feature=['regMonth','regYear','label','adcode','regDate']+group_list
for col in drop_feature:
    feature_cols.remove(col)

print(model_data.shape)
print(province_data.shape)
print(bodyType_data.shape)

print('----------------------------')

model_cols=model_data.columns.tolist()
province_cols=province_data.columns.tolist()
bodyType_cols=bodyType_data.columns.tolist()

print(model_cols)

for col in feature_cols:
    model_cols.remove(col)
    province_cols.remove(col)
    bodyType_cols.remove(col)

test_cols=test_data.columns.tolist()

model_data.regDate=pd.to_datetime(model_data.regDate)
province_data.regDate=pd.to_datetime(province_data.regDate)
bodyType_data.regDate=pd.to_datetime(bodyType_data.regDate)

model_test_data=test_data.merge(model_data,'left',on=model_cols)
province_test_data=test_data.merge(province_data,'left',on=province_cols)
bodyType_test_data=test_data.merge(bodyType_data,'left',on=bodyType_cols)

print(model_test_data.head())

model_test_data.to_csv(folder+'model_test_data.csv',index=None)
province_test_data.to_csv(folder+'province_test_data.csv',index=None)
bodyType_test_data.to_csv(folder+'bodyType_test_data.csv',index=None)























