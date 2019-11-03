#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/3 上午9:25
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

#read
data_folder='data'

train_data=pd.read_csv(os.path.join(data_folder,'all_train_context_data.csv'))
test_data=pd.read_csv(os.path.join(data_folder,'evaluation_public.csv'))

train_data.regDate=pd.to_datetime(train_data['regDate'])
test_data['regDate']=pd.to_datetime([str(year)+'/'+str(month) for month,year in zip(test_data.regMonth,test_data.regYear)])

train_data.sort_values(['regDate'],inplace=True)
test_data.sort_values(['regDate'],inplace=True)

# train_data.set_index(['regDate'],inplace=True)
# test_data.set_index(['regDate'],inplace=True)

feature_cols=train_data.columns.tolist()
group_list=['model','province','bodyType']
drop_feature=['regMonth','regYear','label','adcode']+group_list
for col in drop_feature:
    feature_cols.remove(col)


#overlooka
# print(train_data.columns.tolist())
# print(test_data.columns.tolist())
# print(feature_cols)


'''
generator feature information:  model
                                province
                                bodyType
                                
'''

#generator new data,year iter
last_year_1=pd.date_range('2016-7-1','2016-12-1',freq='m')
last_year_2=pd.date_range('2016-8-1','2017-1-1',freq='m')
last_year_3=pd.date_range('2016-9-1','2017-2-1',freq='m')
last_year_4=pd.date_range('2016-10-1','2017-3-1',freq='m')

last_label_1=pd.to_datetime('2017-1')
last_label_2=pd.to_datetime('2017-2')
last_label_3=pd.to_datetime('2017-3')
last_label_4=pd.to_datetime('2017-4')

feature_year_1=pd.date_range('2017-7-1','2017-12-1',freq='m')
feature_year_2=pd.date_range('2017-8-1','2018-1-1',freq='m')
feature_year_3=pd.date_range('2017-9-1','2018-2-1',freq='m')
feature_year_4=pd.date_range('2017-10-1','2018-3-1',freq='m')

feature_label_1=pd.to_datetime('2018-1')
feature_label_2=pd.to_datetime('2018-2')
feature_label_3=pd.to_datetime('2018-3')
feature_label_4=pd.to_datetime('2018-4')

year_percent=[64,64,32,16,8,4,2]
last_label_list=[last_label_1,last_label_2,last_label_3,last_label_4]
last_year_list=[last_year_1,last_year_2,last_year_3,last_year_4]
feature_label_list=[feature_label_1,feature_label_2,feature_label_3,feature_label_4]
feature_year_list=[feature_year_1,feature_year_2,feature_year_3,feature_year_4]


def generator_new_data(need_data,month):

    last_label=last_label_list[month-1]
    feature_label=feature_label_list[month-1]
    last_year=last_year_list[month-1]
    feature_year=feature_year_list[month-1]

    #ruler information
    last_year=[pd.to_datetime(str(last_year[i].year) + '-' + str(last_year[i].month))for i in range(len(last_year))]
    feature_year=[pd.to_datetime(str(feature_year[i].year) + '-' + str(feature_year[i].month))for i in range(len(feature_year))]

    #5年数据　22cow正确年份的数据　29columns
    percent_data=[(need_data.loc[last_label].values)/(need_data.loc[last_year[i]].values)/regulation for i,regulation in zip(range(len(last_year)),year_percent)]
    new_data=np.array([need_data.loc[feature_year[i]].values*percent_data[i] for i in range(len(feature_year))])

    context_data=np.zeros_like(new_data[0])
    for context_pirce in new_data:
        context_data+=context_pirce

    context_data=context_data.reshape((-1,1))
    need_cols=feature_cols.copy()
    need_cols.remove('regDate')
    need_data=DataFrame(columns=np.array(need_cols),data=context_data.T)
    need_data['regDate']=feature_label


    return need_data

def batch_generator_data(data):
    need_data = data[feature_cols]
    model_context_data = []
    for month in [1,2,3,4]:
        need_data = need_data.groupby('regDate').mean()

        if model_context_data!=[]:
            new_year_data=model_context_data[-1].set_index('regDate')
            need_data=pd.concat([need_data,new_year_data])

        context_data = generator_new_data(need_data, month)
        model_context_data.append(context_data)

    model_context_data=pd.concat(model_context_data)

    return model_context_data

def get_new_data(group_list,feature_cols,group_type):
    group_list.remove(group_type)

    feature_cols_=[]
    for col in group_list:
        if col !=group_type:
            feature_cols_.append(col)

    feature_cols_+=feature_cols

    group_type_list=train_data[group_type].unique().tolist()
    # print(group_type_list)
    by_model = train_data.groupby([group_type])

    context = []
    i=1

    for group_ in group_type_list:
        context_data = batch_generator_data(DataFrame(by_model.get_group(group_)))
        context_data[group_type]=group_
        context.append(context_data)

    context = pd.concat(context)
    # new_feature_cols=[group_type+'_'+col for col in feature_cols_]
    # context.rename(columns={col:new_col for col,new_col in zip(feature_cols_,new_feature_cols)},inplace=True)

    return context

model_context=get_new_data(group_list,feature_cols,'model')
bodyType_context=get_new_data(group_list,feature_cols,'bodyType')
province_context=get_new_data(group_list,feature_cols,'province')

model_context.to_csv('data/model_context.csv',index=None)
bodyType_context.to_csv('data/bodyType_context.csv',index=None)
province_context.to_csv('data/province_context.csv',index=None)




