# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-10-19 下午8:22
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
train = pd.read_csv('./data/train_sales_data.csv')
test = pd.read_csv('./data/evaluation_public.csv')
train16 = train[train['regYear'] == 2016][['adcode', 'model', 'regMonth', 'salesVolume']]
train17 = train[train['regYear'] == 2017][['adcode', 'model', 'regMonth', 'salesVolume']]

#按省份车型16年的均值
df16 = train16.groupby(['adcode', "model"], as_index=False)['salesVolume'].agg({"16mean": 'mean'}) # 按省份和车型统计均值
#按省份车型17年的均值
df17 = train17.groupby(['adcode', "model"], as_index=False)['salesVolume'].agg({"17mean": 'mean'}) # 按省份和车型统计均值

df = pd.merge(df17, df16, on=['adcode', 'model'], how='inner')

df['factor'] = df['17mean'] / df['16mean'] # 17年均值除以16年均值得到趋势因子
# 取出16年12月，17年1,2,3,4,5月，共6个月
df = pd.merge(df, train16[train16['regMonth'] == 12][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'], how='left').rename(columns={'salesVolume': 0})
for m in range(1, 6):
    df = pd.merge(df, train17[train17['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'], how='left').rename(columns={'salesVolume': m})
result_df = pd.DataFrame()
temp_df = df[['adcode', 'model']].copy()
for m in range(1, 5):
    # 预测为上一年的上一个月，同一个月，下一个月的加权,再乘以趋势因子
    temp_df['forecastVolum'] = (df[m - 1].values * 0.25 + df[m].values * 0.5 + df[m + 1].values * 0.25) * df['factor']
    temp_df['regMonth'] = m
    result_df = result_df.append(temp_df, ignore_index=True, sort=False)
    test = pd.merge(test[['id', 'adcode', 'model', 'regMonth']], result_df, how='left', on=['adcode', 'model', 'regMonth'])
    test.loc[test['forecastVolum'] < 0, ['forecastVolum']] = 0
    test[['id', 'forecastVolum']].round(0).astype(int).to_csv('./result/sub.csv', encoding='utf8', index=False)