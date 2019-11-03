# _*_ coding:utf-8 _*_
import pandas as pd
import numpy as np
from pandas import DataFrame
import gc

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)


train_folder='../train/'
test_folder='../test/'
submit_folder='../submit/'

#read_data
sales_data=pd.read_csv(train_folder+'train_sales_data.csv')
search_data=pd.read_csv(train_folder+'train_search_data.csv')
item_context=pd.read_csv(train_folder+'train_user_reply_data.csv')
test_data=pd.read_csv(test_folder+'evaluation_public.csv')

#overlook
'''
all:['adcode' 'bodyType' 'carCommentVolum' 'forecastVolum' 'id' 'model'
 'newsReplyVolum' 'popularity' 'province' 'regMonth' 'regYear'
 'salesVolume']

sales_data: ['province', 'adcode', 'model', 'bodyType', 'regYear', 'regMonth', 'salesVolume']
search_data: ['province', 'adcode', 'model', 'regYear', 'regMonth', 'popularity']
item_context: ['model', 'regYear', 'regMonth', 'carCommentVolum', 'newsReplyVolum']
test_data: ['id', 'province', 'adcode', 'model', 'regYear', 'regMonth', 'forecastVolum']
'''

#补全日期
def generator_date(data):
    date_list=[]
    for year,month in zip(data.regYear.tolist(),data.regMonth.tolist()):
        date=str(year)+'-'+str(month)
        date_list.append(date)
    data['regDate']=date_list
    data.regDate=pd.to_datetime(data.regDate)

    return data
sales_data=generator_date(sales_data)
test_data=generator_date(test_data)

def merge_data(data1,data2):
    on_cols=list(set(data1.columns.tolist())&set(data2.columns.tolist()))
    data1=data1.merge(data2,'left',on=on_cols)
    return data1

#组合数据

#组合训练集和测试集
body_data=sales_data[['province', 'adcode', 'model', 'bodyType']]
test_data=test_data.merge(body_data,'left',on=['province', 'adcode', 'model'])
sales_data.rename(columns={'salesVolume':'label'},inplace=True)
test_data.rename(columns={'forecastVolum':'label'},inplace=True)
test_data.label=test_data.label.fillna(-1)
train_test_data=pd.concat([sales_data,test_data])

#训练集和背景组合
sales_data=sales_data.merge(search_data,'left',on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
train_context_data=sales_data.merge(item_context,'left',on=['model', 'regYear', 'regMonth'])

'''
ad_except_cols=['regDate','model','bodyType']
except of ad   all_bodyType-Month
               bodyType-Month
car_volume:news_volume=4:1
amplification_factor=1000
'''
car_weight=0.8
new_weight=0.2
amplification_factor=1

ad_except_cols=['regMonth','regYear','regDate','model','bodyType','carCommentVolum', 'newsReplyVolum']
ad_except_data=train_context_data[ad_except_cols]
ad_except_data=ad_except_data.drop_duplicates(subset=ad_except_data.columns.tolist(),keep='first')
ad_except_data.set_index(['regDate'],inplace=True)
all_bodyType_Month=DataFrame(ad_except_data[['carCommentVolum','newsReplyVolum']].resample('M').sum())
all_bodyType_Month.reset_index(inplace=True)
all_bodyType_Month['regYear']=all_bodyType_Month['regDate'].dt.year.tolist()
all_bodyType_Month['regMonth']=all_bodyType_Month['regDate'].dt.month.tolist()
del all_bodyType_Month['regDate']

all_bodyType_Month.rename(columns={'carCommentVolum':'all_body_Month_carVolum','newsReplyVolum':'all_body_Month_newVolum'},inplace=True)

bodyType_Month=[]
by_bodyType=ad_except_data.groupby(['bodyType'])
for bodyType in ad_except_data.bodyType.unique().tolist():
    by_bodyType_data=DataFrame(by_bodyType.get_group(bodyType))
    bodyType_data=DataFrame(by_bodyType_data[['carCommentVolum','newsReplyVolum']].resample('M').sum())
    bodyType_data.reset_index(inplace=True)
    bodyType_data['bodyType']=bodyType
    bodyType_Month.append(bodyType_data)
bodyType_Month=pd.concat(bodyType_Month)
bodyType_Month['regYear']=bodyType_Month['regDate'].dt.year.tolist()
bodyType_Month['regMonth']=bodyType_Month['regDate'].dt.month.tolist()
del bodyType_Month['regDate']

bodyType_Month.rename(columns={'carCommentVolum':'body_Month_carVolum','newsReplyVolum':'body_Month_newVolum'},inplace=True)

ad_except_data.reset_index(inplace=True)
ad_except_data=ad_except_data.merge(all_bodyType_Month,'left',on=['regYear','regMonth'])
ad_except_data=ad_except_data.merge(bodyType_Month,'left',on=['regYear','regMonth','bodyType'])

baseCar=np.array(ad_except_data.carCommentVolum.tolist())
baseNew=np.array(ad_except_data.newsReplyVolum.tolist())

all_body_car=np.array(ad_except_data.all_body_Month_carVolum.tolist())
all_body_New=np.array(ad_except_data.all_body_Month_newVolum.tolist())

body_car=np.array(ad_except_data.body_Month_carVolum.tolist())
body_new=np.array(ad_except_data.body_Month_newVolum.tolist())

base_all_body_car=(baseCar/all_body_car)*amplification_factor
base_all_body_New=(baseNew/all_body_New)*amplification_factor
base_body_car=(baseCar/body_car)*amplification_factor
base_body_New=(baseNew/body_new)*amplification_factor

all_body_except=base_all_body_car*car_weight+base_all_body_New*new_weight
body_except=base_body_car*car_weight+base_body_New*new_weight

ad_except_data['base_all_body_car']=base_all_body_car.tolist()
ad_except_data['base_all_body_New']=base_all_body_New.tolist()
ad_except_data['base_body_car']=base_body_car.tolist()
ad_except_data['base_body_new']=base_body_New.tolist()
ad_except_data['all_body_except']=all_body_except.tolist()
ad_except_data['body_except']=body_except.tolist()

train_context_data=merge_data(train_context_data,ad_except_data)

'''
use_cols=['province','regDate','popularity','all_body_except','body_except']
popular of model: all_province-month
                  province-month
'''

use_cols=['province','regDate','regMonth','regYear','popularity','all_body_except','body_except']
popular_data=train_context_data[use_cols]
popular_data.set_index(['regDate'],inplace=True)

#all_province_month
all_province_month=DataFrame(popular_data['popularity'].resample('M').sum())
all_province_month.reset_index(inplace=True)
all_province_month.rename(columns={'popularity':'all_province_month'},inplace=True)
all_province_month['regYear']=all_province_month.regDate.dt.year
all_province_month['regMonth']=all_province_month.regDate.dt.month

del all_province_month['regDate']

#province_month
province_month=[]
by_province=popular_data.groupby(['province'])
for province in popular_data.province.unique().tolist():
    by_province_data=DataFrame(by_province.get_group(province))
    province_month_data=DataFrame(by_province_data['popularity'].resample('M').sum())
    province_month_data.rename(columns={'popularity':'province_month'},inplace=True)
    province_month_data.reset_index(inplace=True)
    province_month_data['province']=province
    province_month.append(province_month_data)
province_month=pd.concat(province_month)
province_month['regYear']=province_month.regDate.dt.year
province_month['regMonth']=province_month.regDate.dt.month

del province_month['regDate']

popular_data.reset_index(inplace=True)
popular_data=merge_data(popular_data,all_province_month)
popular_data=merge_data(popular_data,province_month)

basepopu=popular_data.popularity.tolist()
allpopu=popular_data.all_province_month.tolist()
provincepopu=popular_data.province_month.tolist()

all_province_popular=np.array(basepopu)/np.array(allpopu)
province_popular=np.array(basepopu)/np.array(provincepopu)

popular_data['all_province_popularity']=all_province_popular.tolist()
popular_data['province_popularity']=province_popular.tolist()

train_context_data=merge_data(train_context_data,popular_data)

'''
popularity:ad_except=3:2
car_attention=popularity+ad_except
'''

ad_except_weight=0.4
popularity_weight=0.6

all_province_all_body_attention=np.array(train_context_data.all_province_popularity.tolist())*popularity_weight+np.array(train_context_data.all_body_except.tolist())*ad_except_weight
all_province_body_attention=np.array(train_context_data.all_province_popularity.tolist())*popularity_weight+np.array(train_context_data.body_except.tolist())*ad_except_weight
province_all_body_attention=np.array(train_context_data.province_popularity.tolist())*popularity_weight+np.array(train_context_data.all_body_except.tolist())*ad_except_weight
province_body_attention=np.array(train_context_data.province_popularity.tolist())*popularity_weight+np.array(train_context_data.body_except.tolist())*ad_except_weight

train_context_data['all_province_all_body_attention']=all_province_all_body_attention.tolist()
train_context_data['all_province_body_attention']=all_province_body_attention.tolist()
train_context_data['province_all_body_attention']=province_all_body_attention.tolist()
train_context_data['province_body_attention']=province_body_attention.tolist()

'''
use_cols=['regDate','regMonth','regYear','label','bodyType','model','province']
model sales of  all_bodyType-all_province-Month
                all_bodyType-province-Month

                bodyType-all_province-Month
                bodyType-province-Month

model sales of all_bodyType-all_province-Month
'''
use_cols=['regDate','regMonth','regYear','label','bodyType','model','province']
sales_relation_data=train_context_data[use_cols]
sales_relation_data.set_index(['regDate'],inplace=True)

#all_bodyType-all_province-Month
all_bodyTyPe_all_province_Month=DataFrame(sales_relation_data['label'].resample('M').sum())
all_bodyTyPe_all_province_Month.reset_index(inplace=True)
all_bodyTyPe_all_province_Month['regYear']=all_bodyTyPe_all_province_Month.regDate.dt.year
all_bodyTyPe_all_province_Month['regMonth']=all_bodyTyPe_all_province_Month.regDate.dt.month
all_bodyTyPe_all_province_Month.rename(columns={'label':'all_bodyType_all_province_Month_label'},inplace=True)

del all_bodyTyPe_all_province_Month['regDate']

#all_bodyType-province-Month
all_bodyType_province_Month=[]

by_province_sales=sales_relation_data.groupby(['province'])
for province in sales_relation_data.province.unique().tolist():
    by_province_sales_data=DataFrame(by_province_sales.get_group(province))
    province_sales_data=DataFrame(by_province_sales_data['label'].resample('M').sum())
    province_sales_data.reset_index(inplace=True)
    province_sales_data['province']=province
    all_bodyType_province_Month.append(province_sales_data)

all_bodyType_province_Month=pd.concat(all_bodyType_province_Month)
all_bodyType_province_Month['regYear']=all_bodyType_province_Month.regDate.dt.year
all_bodyType_province_Month['regMonth']=all_bodyType_province_Month.regDate.dt.month
all_bodyType_province_Month.rename(columns={'label':'all_bodyType_province_Month_label'},inplace=True)

del all_bodyType_province_Month['regDate']

#bodyType-all_province-Month
bodyType_all_province_Month=[]
by_bodyType_sales=sales_relation_data.groupby(['bodyType'])

for bodyType in sales_relation_data['bodyType'].unique().tolist():
    by_bodyType_sales_data=DataFrame(by_bodyType_sales.get_group(bodyType))
    bodyType_sales_data=DataFrame(by_bodyType_sales_data['label'].resample('M').sum())
    bodyType_sales_data.reset_index(inplace=True)
    bodyType_sales_data['bodyType']=bodyType
    bodyType_all_province_Month.append(bodyType_sales_data)

bodyType_all_province_Month=pd.concat(bodyType_all_province_Month)
bodyType_all_province_Month['regYear']=bodyType_all_province_Month.regDate.dt.year
bodyType_all_province_Month['regMonth']=bodyType_all_province_Month.regDate.dt.month
bodyType_all_province_Month.rename(columns={'label':'bodyType_all_province_Month_label'},inplace=True)

del bodyType_all_province_Month['regDate']

#bodyType-province-Month
bodyType_province_Month=[]

by_bodyType_sales=sales_relation_data.groupby(['bodyType'])
for bodyType in sales_relation_data.bodyType.unique().tolist():
    by_bodyType_sales_data=DataFrame(by_bodyType_sales.get_group(bodyType))
    by_province_body=by_bodyType_sales_data.groupby(['province'])
    for province in sales_relation_data.province.unique().tolist():
        by_province_body_data=DataFrame(by_province_body.get_group(province))
        province_body_data=DataFrame(by_province_body_data['label'].resample('M').sum())
        province_body_data['bodyType']=bodyType
        province_body_data['province']=province
        province_body_data.reset_index(inplace=True)
        bodyType_province_Month.append(province_body_data)

bodyType_province_Month=pd.concat(bodyType_province_Month)
bodyType_province_Month['regYear']=bodyType_province_Month.regDate.dt.year
bodyType_province_Month['regMonth']=bodyType_province_Month.regDate.dt.month
bodyType_province_Month.rename(columns={'label':'bodyType_province_Month_label'},inplace=True)

del bodyType_province_Month['regDate']

sales_relation_data=merge_data(sales_relation_data,all_bodyTyPe_all_province_Month)
sales_relation_data=merge_data(sales_relation_data,all_bodyType_province_Month)
sales_relation_data=merge_data(sales_relation_data,bodyType_all_province_Month)
sales_relation_data=merge_data(sales_relation_data,bodyType_province_Month)

base_label=np.array(sales_relation_data.label.tolist())
month_label=np.array(sales_relation_data.all_bodyType_all_province_Month_label.tolist())
province_label=np.array(sales_relation_data.all_bodyType_province_Month_label.tolist())
bodyType_label=np.array(sales_relation_data.bodyType_all_province_Month_label.tolist())
province_bodyType_label=np.array(sales_relation_data.bodyType_province_Month_label.tolist())

month_model_weight=base_label/month_label
province_model_weight=base_label/province_label
bodyType_model_weight=base_label/bodyType_label
province_bodyType_model_weight=base_label/province_bodyType_label

sales_relation_data['month_model_weight']=month_model_weight.tolist()
sales_relation_data['province_model_weight']=province_model_weight.tolist()
sales_relation_data['bodyType_model_weight']=bodyType_model_weight.tolist()
sales_relation_data['province_bodyType_model_weight']=province_bodyType_model_weight.tolist()

train_context_data=merge_data(train_context_data,sales_relation_data)


useful_cols=['province', 'adcode', 'model', 'bodyType', 'regYear', 'regMonth', 'label', 'regDate','all_body_except', 'body_except','all_province_all_body_attention', 'all_province_body_attention', 'province_all_body_attention', 'province_body_attention','month_model_weight', 'province_model_weight', 'bodyType_model_weight', 'province_bodyType_model_weight']
train_context_data.to_csv('data/all_train_context_data.csv',index=None)
train_context_data[useful_cols].to_csv('data/useful_trian_context_data.csv',index=None)


