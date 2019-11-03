# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-10-19 上午10:25
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
data=pd.read_csv('df_lgb-10-23.csv')
data[['id','ad_ry_mean']].to_csv('cut1.csv',index=None)
data[['id','md_ry_mean']].to_csv('cut2.csv',index=None)
data[['id','bt_ry_mean']].to_csv('cut3.csv',index=None)
