# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-10-19 下午7:58
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
for month in [25, 26, 27, 28]:
    m_type = 'lgb'

    data_df, stat_feat = get_stat_feature(data)

    num_feat = ['regYear'] + stat_feat
    cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']

    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

    features = num_feat + cate_feat
    print(len(features), len(set(features)))

    sub, val_pred = get_train_model(data_df, month, m_type)
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values
sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
sub.columns = ['id', 'forecastVolum']
sub[['id', 'forecastVolum']].round().astype(int).to_csv('CCF_sales.csv', index=False)