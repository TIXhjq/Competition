# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-10-22 下午4:03
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
import gc
import os
import re

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
data=pd.read_csv('data/train_sales_data.csv')
popu=pd.read_csv('data/train_search_data.csv')
user=pd.read_csv('data/train_user_reply_data.csv')


def cal_same_cols(data1,data2):
    return list(set(data1.columns.tolist())&set(data2.columns.tolist()))

data=data.merge(popu,how='left',on=cal_same_cols(data,popu))
data=data.merge(user,how='left',on=cal_same_cols(data,user))

from sklearn.preprocessing import LabelEncoder
def obj_encoder(data:DataFrame,col_list:list):
    for col in col_list:
        data[col]=LabelEncoder().fit_transform(data[col]).tolist()
    return data

data=obj_encoder(data=data,col_list=['model','bodyType'])


province_rank=pd.read_csv('data/test.csv')
province_rank.columns=province_rank.loc[0].tolist()
pro_adcode=data[['province','adcode']].drop_duplicates()
province_rank.drop(0,inplace=True)
province_rank.rename(columns={'地区':'province','排名':'rank'},inplace=True)
data=data.merge(province_rank,on=['province'])

data.sort_values(['rank'],inplace=True)
province_rank_=data[['adcode','province','rank']].drop_duplicates().sort_values(by=['rank']).reset_index().drop(columns=['index'])
province_rank_['rank']=province_rank_['rank'].astype(int).tolist()
province_rank_.sort_values(by=['rank'],inplace=True)

rich_province=province_rank_[province_rank_['rank']<=6]['adcode'].tolist()
pool_province=province_rank_[province_rank_['rank']>=24]['adcode'].tolist()


province_sum_salesVolume=DataFrame(data.loc[data.adcode.isin(pool_province)].groupby(['regMonth','adcode'])['salesVolume'].sum()).reset_index()
province_sum_salesVolume.rename(columns={'salesVolume':'sum_salesVolume'},inplace=True)
province_model_salesVolume=DataFrame(data.loc[data.adcode.isin(pool_province)].groupby(['regMonth','adcode','model'])['salesVolume'].sum()).reset_index()
province_model_salesVolume=province_model_salesVolume.merge(province_sum_salesVolume,on=['adcode','regMonth'],how='left')

del province_sum_salesVolume
gc.collect()

province_model_salesVolume=DataFrame(province_model_salesVolume.groupby(['adcode','model'])[['salesVolume','sum_salesVolume']].sum()).reset_index()
print(province_model_salesVolume.head())
province_model_salesVolume['model_precent']=province_model_salesVolume['salesVolume']/province_model_salesVolume['sum_salesVolume']
# province_model_salesVolume=DataFrame(province_model_salesVolume.groupby(['adcode','model'])['model_precent'].).reset_index()

print(province_model_salesVolume[province_model_salesVolume['model_precent']>=0.05].model.unique().tolist())

by_adcode=province_model_salesVolume.groupby(['adcode'])
plt.figure()
i=1
for adcode in pool_province:
    x=DataFrame(by_adcode.get_group(adcode)).sort_values(['model_precent'])['model']
    y=DataFrame(by_adcode.get_group(adcode)).sort_values(['model_precent'])['model_precent']

    plt.subplot(3,3,i)
    plt.bar(x,y)
    plt.title(adcode)
    i+=1
plt.show()



        # x=province_model_salesVolume[(province_model_salesVolume['adcode'].isin([adcode]))&(province_model_salesVolume['model']==model)]['regMonth']
        # y=province_model_salesVolume[(province_model_salesVolume['adcode'].isin([adcode]))&(province_model_salesVolume['model']==model)]['model_precent']


data.loc[data.adcode.isin(rich_province)].groupby(['regMonth','adcode'])['salesVolume'].mean()



aim_fea=['salesVolume','popularity','carCommentVolum','newsReplyVolum']
rely_fea=['regMonth']
rely_fea_base=rely_fea+['adcode']
rely_fea_body=rely_fea+['bodyType']
rely_fea_model=rely_fea+['model']
rely_fea_base_body=rely_fea_base+['bodyType']
rely_fea_base_model=rely_fea_base+['model']
#
#
#
#
# data_16=data.loc[data.regYear==2016].groupby(rely_fea_base)[aim_fea].mean().reset_index()
# data.loc[data.regYear==2017].groupby(rely_fea_base)[aim_fea].mean().reset_index()
# print(len(data.adcode.unique().tolist()))
#
# def plot_(data):
#     data_16=data.loc[data.regYear == 2016].groupby(rely_fea_model)[aim_fea].mean().reset_index()
#     data_17=data.loc[data.regYear == 2017].groupby(rely_fea_model)[aim_fea].mean().reset_index()
#
#     plt.figure(figsize=(6,6))
#     by_adcode_16=data_16.groupby(['model'])
#     by_adcode_17=data_17.groupby(['model'])
#
#     from tqdm import tqdm
#
#     for fea in aim_fea:
#         num = 0
#         for adcode in tqdm(data.model.unique().tolist()):
#             num += 1
#             adcode_data_16=DataFrame(by_adcode_16.get_group(adcode))
#             adcode_data_17=DataFrame(by_adcode_17.get_group(adcode))
#
#             x_16=adcode_data_16['regMonth']
#             y_16=adcode_data_16[fea]
#
#             x_17=adcode_data_17['regMonth']
#             y_17=adcode_data_17[fea]
#
#             plt.subplot(10,6,num)

#             plt.plot(x_16,y_16,label='16_data')
#             plt.plot(x_17,y_17,label='17_data')
#             # plt.xlabel('regMonth')
#             # plt.ylabel(fea)
#             # plt.legend()
#             plt.title(adcode)
#         #
#         #     #saleVolume :adcode 110000,500000
#
#         #--------------------
#         #全国数据
#         # x_16=data_16['regMonth']
#         # y_16=data_16[fea]
#         #
#         # x_17=data_17['regMonth']
#         # y_17=data_17[fea]
#         # plt.plot(x_16, y_16, label='16_data')
#         # plt.plot(x_17,y_17,label='17_data')
#         # plt.xlabel('regMonth')
#         # plt.ylabel(fea)
#         #--------------------
#
#         plt.title(fea)
#         plt.show()
#
# plot_(data)
# data.loc[data.regYear==2016].groupby(rely_fea_body)[aim_fea].mean()
# data.loc[data.regYear==2017].groupby(rely_fea_body)[aim_fea].mean()
#
# data.loc[data.regYear==2016].groupby(rely_fea_model)[aim_fea].mean()
# data.loc[data.regYear==2017].groupby(rely_fea_model)[aim_fea].mean()
#
