import pandas as pd
import numpy as np
import time
from pandas import DataFrame
from pandas import timedelta_range
from datetime import timedelta
import matplotlib.pyplot as plt
import gc


def plot_date_salesVolume(title, data):
    title+='-date-sales'
    data.sort_values(['regDate'], inplace=True)
    data.set_index(['regDate'],inplace=True)
    month_salesVolume=DataFrame(data['salesVolume'].resample('M').sum())
    month_salesVolume.reset_index(inplace=True)
    print(month_salesVolume)

    x=month_salesVolume.regDate.tolist()
    y=month_salesVolume.salesVolume.tolist()

    plt.plot(x, y)
    plt.xlabel('Date')
    plt.ylabel('salesVolume')
    plt.title(title)
    plt.show()

sales_data=pd.read_csv('data/train/train_sales_data.csv')

#Date-sales
Year=sales_data.regYear.tolist()
Month=sales_data.regMonth.tolist()

Date=[]
for year,month in zip(Year,Month):
    date=str(year)+'/'+str(month)
    Date.append(date)

sales_data['regDate']=Date
sales_data.regDate=pd.to_datetime(sales_data['regDate'])

del sales_data['regMonth'],sales_data['regYear']
gc.collect()

#date-sales(first)
date_sales=sales_data[['salesVolume','regDate']]
plot_date_salesVolume(title='all_data',data=date_sales)


#province-date-sales(second)
province_list=sales_data.province.unique().tolist()
by_province_data=sales_data.groupby(['province'])

for province in province_list:
    by_province_date_sales_data=DataFrame(by_province_data.get_group(province))[['salesVolume','regDate']]
    plot_date_salesVolume(title=str(province),data=by_province_date_sales_data)

#car_type-date-sales(second)
car_type_list=sales_data.bodyType.unique().tolist()
by_car_type_data=sales_data.groupby(['bodyType'])

for car_type in car_type_list:
    by_car_type_date_sales_data=DataFrame(by_car_type_data.get_group(car_type))[['salesVolume','regDate']]
    plot_date_salesVolume(title=str(car_type),data=by_car_type_date_sales_data)

#car_type-province-date-sales(third)
province_list=sales_data.province.unique().tolist()
by_province_data=sales_data.groupby(['province'])

for province in province_list:
    by_province_date_sales_data=DataFrame(by_province_data.get_group(province))
    by_car_type_by_province_data=by_province_date_sales_data.groupby(['bodyType'])
    for  car_type in car_type_list:
        by_car_type_by_province_date_sales_data=by_car_type_by_province_data.get_group(car_type)[['salesVolume','regDate']]
        plot_date_salesVolume(title=str(car_type)+'-'+str(province),data=by_car_type_by_province_date_sales_data)

#car_model_province-date-sales(forth)
#car_model包含了car_type
province_list = sales_data.province.unique().tolist()
by_province_data = sales_data.groupby(['province'])
car_model_list=sales_data.model.unique().tolist()

for province in province_list:
    by_province_date_sales_data = DataFrame(by_province_data.get_group(province))
    by_car_model_by_province_data = by_province_date_sales_data.groupby(['model'])
    for car_model in car_model_list:
        by_car_model_by_province_date_sales_data = by_car_model_by_province_data.get_group(car_model)[['salesVolume','regDate']]
        plot_date_salesVolume(title=str(car_model)+str(province),data=by_car_model_by_province_date_sales_data)





