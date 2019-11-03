# _*_ coding:utf-8 _*_
import pandas as pd
import numpy as np
from pandas import DataFrame

np.set_printoptions(suppress=True, threshold=np.nan)
pd.set_option('display.max_columns', 10000, 'display.max_rows', 10000)

original_features=['province','adcode','model','bodyType','regYear','regMonth','label','regDate','popularity','carCommentVolum','newsReplyVolum']
first_features=[
    'all_body_Month_carVolum','all_body_Month_newVolum','body_Month_carVolum','body_Month_newVolum','base_all_body_car','base_all_body_New','base_body_car','base_body_new',

    'all_province_month', 'province_month', 'all_province_popularity', 'province_popularity',
    'all_bodyType_all_province_Month_label', 'all_bodyType_province_Month_label', 'bodyType_all_province_Month_label', 'bodyType_province_Month_label'
]
second_features=['all_body_except','body_except','all_province_all_body_attention', 'all_province_body_attention', 'province_all_body_attention', 'province_body_attention','month_model_weight', 'province_model_weight', 'bodyType_model_weight', 'province_bodyType_model_weight']

stable_feature=pd.read_csv('data/stable_feature.csv')
feature_important=pd.read_csv('data/feature_important.csv')

final_feature=feature_important['feature_cols'].tolist()+['id']
test_data=pd.read_csv('data/evaluation_public.csv')
train=pd.read_csv('data/all_train_context_data.csv')

# stable=DataFrame(stable_feature.std())
# stable.reset_index(inplace=True)
# stable.columns=['feature_cols','std']
# stable.sort_values(by=['std'],inplace=True)
# print(stable.)


train['price_price']=(train['all_body_except']*train['province_all_body_attention']).tolist()


final_train=train
drop_feature=['regMonth','regYear','model','label','adcode']
feature_cols=final_train.columns.tolist()
for feature in drop_feature:
    feature_cols.remove(feature)



all_test_context=[]
by_model=final_train.groupby(['model'])
for model in final_train.model.unique().tolist():
    by_model_data=DataFrame(by_model.get_group(model))
    by_province=by_model_data.groupby(['province'])
    for province in by_model_data.province.unique().tolist():
        by_province_data=DataFrame(by_province.get_group(province))
        by_month=by_province_data.groupby(['regMonth'])
        for month in by_province_data.regMonth.unique().tolist():
            by_month_data=DataFrame(by_month.get_group(month)).sort_values(by=['regDate'])
            aim_data=by_month_data[feature_cols]


            aim_=DataFrame(aim_data.mean())
            feature_mean=np.array(aim_[0].tolist()).reshape((1,-1))
            feature_col=aim_.index.tolist()
            test_=DataFrame(columns=feature_col,data=feature_mean)
            test_['regMonth']=month
            test_['model']=model
            test_['province']=province
            all_test_context.append(test_)
all_test_context=pd.concat(all_test_context)

on_cols=list(set(all_test_context.columns.tolist())&set(test_data.columns.tolist()))
test_data=test_data.merge(all_test_context,'left',on=on_cols)
# test_data=test_data[final_feature]
test_data['forecastVolum']=-1
test_data.to_csv('data/test_data.csv',index=True)
