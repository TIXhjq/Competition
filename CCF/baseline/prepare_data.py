# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-10-19 下午3:50
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
import gc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
folder='data'

train_sales_data=pd.read_csv(os.path.join(folder,'train_sales_data.csv'))
train_search_data=pd.read_csv(os.path.join(folder,'train_search_data.csv'))
train_user_reply_data=pd.read_csv(os.path.join(folder,'train_user_reply_data.csv'))
test_data=pd.read_csv(os.path.join(folder,'evaluation_public.csv'))

spring_shut_up_business=7
year_16_month_days=[31,29-spring_shut_up_business,31,30,31,30,31,31,30,31,30,31]
year_17_month_days=[31-spring_shut_up_business,28,31,30,31,30,31,31,30,31,30,31]
year_18_month_days=[31,28-spring_shut_up_business,31,30]
year_month_days=[year_16_month_days,year_17_month_days,year_18_month_days]

user_16_holiday=[3,0,0,4,2,3,0,0,3,7,0,1]
user_17_holiday=[2,0,0,5,4,0,0,0,0,7,0,2]
user_18_holiday=[1,0,0,5]

#拼接特征
train_sales_data=train_sales_data.merge(train_search_data,how='left',on=['province','adcode','model','regYear','regMonth'])
train_sales_data=train_sales_data.merge(train_user_reply_data,how='left',on=['model','regYear','regMonth'])

#计算日收益
def day_month_transform(data,year_list=[2016,2017],aim_fea=['salesVolume','popularity'],to_days=True):
    for year in year_list:
        month_days=year_month_days[year - 2016]
        div_=[month_days[month-1] for month in data.loc[data.regYear==year,'regMonth']]
        if to_days:
            data.loc[train_sales_data.regYear==year,aim_fea]/=np.array(div_*2).reshape((np.array(div_).shape[0],len(aim_fea)))
        else:
            data.loc[train_sales_data.regYear==year,aim_fea]*=np.array(div_*2).reshape((np.array(div_).shape[0],len(aim_fea)))

    return data

train_sales_data=day_month_transform(train_sales_data)

#构建用户假期特征
train_sales_data['holiday']=train_sales_data.regMonth.tolist()
test_data['holiday']=test_data.regMonth.tolist()
train_sales_data.loc[train_sales_data.regYear==2016,'holiday']=train_sales_data.loc[train_sales_data.regYear==2016,'holiday'].replace(to_replace=dict(zip(range(1,13),user_16_holiday))).tolist()
train_sales_data.loc[train_sales_data.regYear==2017,'holiday']=train_sales_data.loc[train_sales_data.regYear==2017,'holiday'].replace(to_replace=dict(zip(range(1,13),user_17_holiday))).tolist()
test_data.loc[test_data.regYear==2018,'holiday']=test_data.loc[test_data.regYear==2018,'holiday'].replace(to_replace=dict(zip(range(1,5),user_18_holiday))).tolist()
print(test_data.head())
test_data.to_csv('data/test.csv',index=None)

#数据基本处理
train_sales_data.drop(columns=['province'],inplace=True)
train_sales_data.bodyType=LabelEncoder().fit_transform(train_sales_data.bodyType)
tool_data=train_sales_data[['model','adcode','bodyType']]
test_data=test_data.merge(tool_data,how='left',on=['model','adcode'])
del tool_data
train_sales_data['mt']=(train_sales_data.regYear-2016)*12+train_sales_data.regMonth
test_data['mt']=(test_data.regYear-2016)*12+test_data.regMonth

#generate robust fea
def cal_robust_fea(data:DataFrame,rely_fea:list,aim_fea:list):
    cal_format=['sum','mean']
    fea_cols_list=[]
    for cal in tqdm(cal_format,desc='generator_robust_feature'):
        ad_body_mt_popular=data.groupby(rely_fea)[aim_fea].agg(cal).reset_index()
        fea_cols={col:(cal+'_'+'_'.join(rely_fea)+'_'+col) for col in aim_fea}
        ad_body_mt_popular.rename(columns=fea_cols,inplace=True)
        data=data.merge(ad_body_mt_popular,how='left',on=rely_fea)
        fea_cols_list+=fea_cols.values()

    return data,fea_cols_list

rely_fea_list=[['adcode','model','mt'],['adcode','bodyType','mt']]
               # ['adcode','mt'],['bodyType','mt']]
aim_fea=['popularity','salesVolume']
fea_cols_list=[]

for rely_fea in rely_fea_list:
    train_sales_data,fea_cols=cal_robust_fea(data=train_sales_data,rely_fea=rely_fea,aim_fea=aim_fea)
    fea_cols_list+=fea_cols


#win_fea
def generate_win_fea(data,win_list:list,rely_fea:list,aim_fea:list):
    win_col_list=[]
    for win in tqdm(win_list,desc='generator_win_feature'):
        win_fea=data.groupby(rely_fea)[aim_fea].rolling(win).mean().reset_index()
        win_cols={aim :'win_'+str(win)+"_"+'_'.join(rely_fea)+'_'+aim for aim in aim_fea}
        win_fea.rename(columns=win_cols,inplace=True)
        win_col_list.append(win_cols)
        data=data.merge(win_fea,on=rely_fea,how='left')
        del win_fea
        gc.collect()

    return data,win_col_list

train_sales_data.sort_values(by=['mt'],inplace=True)
train_sales_data,win_cols=generate_win_fea(data=train_sales_data,win_list=[3],rely_fea=['adcode','bodyType'],aim_fea=fea_cols_list)

generate_fea_cols=win_cols+fea_cols_list
train_sales_data['win_date']=train_sales_data.mt+12

del train_search_data,train_user_reply_data,train_sales_data['regYear'],test_data['province'],test_data['forecastVolum']
gc.collect()

test_data.rename(columns={'mt':'win_date'},inplace=True)
test_data=test_data.merge(train_sales_data,on=['adcode','bodyType','win_date'])
print(train_sales_data.head())
# print(test_data.head())
# lgb_model = lgb.LGBMRegressor(
#         num_leaves=40, reg_alpha=1, reg_lambda=0.1, objective='mse',
#         max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2048,
#         n_estimators=8000, subsample=0.8, colsample_bytree=0.8)
#
# lgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
#               categorical_feature=cate_feat, early_stopping_rounds=100, verbose=300)
# data['pred_label'] = np.e ** lgb_model.predict(data[features])
# model = lgb_model
