# _*_ coding:utf-8 _*_
import pandas as pd
import numpy as np
from pandas import DataFrame

test_data=pd.read_csv('data/evaluation_public.csv')
all_train_context_data=pd.read_csv('data/all_train_context_data.csv')

stable_feature=[]
by_model=all_train_context_data.groupby(['model'])
for model in all_train_context_data.model.unique().tolist():
    by_model_data=DataFrame(by_model.get_group(model))
    by_province=by_model_data.groupby(['province'])
    for province in by_model_data.province.unique().tolist():
        by_province_data=DataFrame(by_province.get_group(province))
        by_month=by_province_data.groupby(['regMonth'])
        for month in by_province_data.regMonth.unique().tolist():
            by_month_data=DataFrame(by_month.get_group(month))
            std_data=DataFrame(by_month_data.std())
            feature_=std_data.index.tolist()
            std_=np.array(std_data[0].tolist()).reshape((1,-1))
            std_information=DataFrame(columns=feature_,data=std_)
            std_information['month']=month
            std_information['province']=province
            std_information['model']=model
            stable_feature.append(std_information)

stable_feature=pd.concat(stable_feature)
stable_feature.to_csv('data/stable_feature.csv',index=None)
print(stable_feature.head())