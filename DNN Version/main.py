# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:04:25 2026

@author: Administrator
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TSMixerx, NHITS
from neuralforecast.losses.pytorch import MSE, MAE
from sklearn.metrics import root_mean_squared_error

from logger import init_logger

DATA_PATH = 'data'
DATE_FEATURES = ['日期', '当日 MAXV 对应时间', '当日 MINV 对应时间']
NUM_FEATURES = ['充电功率/MW', '当日平均充电功率/MW', '当日最大充电功率/MW', '当日 V 最小值/MW', 
                '充电时长/h', '当日所有 S 的均值/h', '当日 S 的最大值/h', '当日 S 的最小值/h', 
                '当日最大充电时长跨度/h']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体

def set_random_seed():
    import torch
    import random
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # 如果使用GPU，设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_ts_data():

    ts_data = pd.read_csv(os.path.join(DATA_PATH, 'A榜-充电站充电负荷训练数据.csv'), encoding='GBK')
    ts_data.drop(index=ts_data.index[0], inplace=True)
    
    for col in ts_data.columns:
        if col in DATE_FEATURES:
            ts_data[col] = pd.to_datetime(ts_data[col])
        elif col in NUM_FEATURES:
            ts_data[col] = pd.to_numeric(ts_data[col])
            
    return ts_data
    
def format_day(ts_data):
    ts_data_day = ts_data.copy()
    ts_data_day['日期'] = ts_data_day['日期'].dt.normalize()
    ts_data_day = ts_data_day.drop(columns=['充电功率/MW', '充电时长/h']).drop_duplicates()
    return ts_data_day
    
    
def figure(ts_data, start_date='2024/1/1 0:00:00', end_date='2024/10/31 23:45:00'):

    ts_data = ts_data[(ts_data['日期'] >= pd.to_datetime(start_date))&(ts_data['日期'] <= pd.to_datetime(end_date))]
    ts_data_day = format_day(ts_data)
    fig = plt.figure(figsize=(10, 6), dpi=200)
    # fig.suptitle('{} GISID[{}]'.format(title, gisid))
    gs = GridSpec(2, 1)
    
    # 绘制功率与充电时长（15分钟）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.grid(True, alpha=0.7)

    ax1.set_xlabel('时间')
    ax1.set_ylabel('充电功率/MW')
    ax1.plot(ts_data['日期'], ts_data['充电功率/MW'], label='充电功率/MW')
    ax1.legend(loc='upper left')
    
    ax1_ = ax1.twinx()  # 共享 x 轴
    ax1_.set_ylabel('充电时长/h')
    ax1_.plot(ts_data['日期'], ts_data['充电时长/h'], label='充电时长/h', alpha=0.8, c='#ff7f0e')
    ax1_.legend(loc='upper right')
    
    # 绘制功率与充电时长（日）
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.grid(True, alpha=0.7)

    ax1.set_xlabel('时间')
    ax1.set_ylabel('当日最大充电功率/MW')
    ax1.plot(ts_data_day['日期'], ts_data_day['当日最大充电功率/MW'], label='当日最大充电功率/MW')
    ax1.legend(loc='upper left')
    
    ax1_ = ax1.twinx()  # 共享 x 轴
    ax1_.set_ylabel('当日 S 的最大值/h')
    ax1_.plot(ts_data_day['日期'], ts_data_day['当日 S 的最大值/h'], label='当日 S 的最大值/h', alpha=0.8, c='#ff7f0e')
    ax1_.legend(loc='upper right')
    
    plt.show()
    
def figure_pred(train, test, pred):
    train = train[-96 * 3:]
    
    plt.figure(figsize=(10, 3), dpi=200)
    plt.grid(True, alpha=0.7)
    
    plt.plot(train['ds'], train['y'], label='训练集')
    plt.plot(test['ds'], test['y'], label='测试集')
    plt.plot(pred['ds'], pred['y'], label='预测集')
    
    plt.legend(loc='upper left')
    plt.savefig(f'figures/{MODEL}_{EPOCH}_{BATCH_SIZE}.png')
    plt.show()
    
# 转换时间为时分秒
def time_conver(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 2)
    
    if hours == 0 and minutes == 0:
        time = '{}s'.format(seconds)
    elif hours == 0:
        time = '{}m:{}s'.format(minutes, seconds)
    else:
        time = '{}h:{}m:{}s'.format(hours, minutes, int(seconds))
    
    # return ' {:02d}h {:02d}m {:0}s'.format(hours, minutes, seconds)
    return time
    
    
class Forecaster:
    def __init__(self):
        self.pred_points = PRED_POINTS
        self.input_points = INPUT_POINTS
        self.scaler_type = NORMAL_METHOD
        self.init_model()
        
    def init_model(self, exog_list = None):
        
        if LOSS == 'MSE':
            self.loss = MSE()
        elif LOSS == 'MAE':
            self.loss = MAE()
            
        # 预测模型
        if MODEL == 'TSMixerx':
            ts_mixer_model = TSMixerx(
                h=self.pred_points, # 预测的时间点
                input_size=self.input_points, # 输入天数                              
                futr_exog_list=exog_list,  # 未来已知外生变量
                n_series=1,
                scaler_type=self.scaler_type, # 数据标准化方法
                max_steps=EPOCH, # 训练迭代轮数
                val_check_steps=5, # 验证轮数
                learning_rate=LR, # 学习率
                loss=self.loss, 
                valid_loss=self.loss, # 损失函数
                batch_size=BATCH_SIZE, # 输入batch大小
                accelerator='cpu', # 训练设备
                devices=1, # 设备数量
                random_seed=SEED, # 随机种子
                #################################################
                enable_model_summary=False, # 关闭训练过程输出
                enable_progress_bar=False,
                logger=False, 
                enable_checkpointing=False)
            
        elif MODEL == 'NHITS':
            ts_mixer_model = NHITS(
                h=self.pred_points, # 预测的时间点
                input_size=self.input_points, # 输入天数                              
                futr_exog_list=exog_list,  # 未来已知外生变量
                scaler_type=self.scaler_type, # 数据标准化方法
                max_steps=EPOCH, # 训练迭代轮数
                val_check_steps=5, # 验证轮数
                learning_rate=LR, # 学习率
                loss=self.loss, 
                valid_loss=self.loss, # 损失函数
                batch_size=BATCH_SIZE, # 输入batch大小
                accelerator='cpu', # 训练设备
                devices=1, # 设备数量
                random_seed=SEED, # 随机种子
                #################################################
                enable_model_summary=False, # 关闭训练过程输出
                enable_progress_bar=False,
                logger=False, 
                enable_checkpointing=False)

        # 预测器
        self.nf = NeuralForecast(models=[ts_mixer_model], freq='15min') # D 代表以天为预测单位
                                 
        
    def format_data(self, ts_data):
        ts_data.rename(columns={'日期': 'ds',
                                '充电功率/MW': 'y'
                                }, inplace=True)
        ts_data = ts_data[['ds', 'y']]
        ts_data['unique_id'] = 'power'
        return ts_data
    
    def create_future_df(self, split_date):
        
        future_df = pd.DataFrame()
        date_range = pd.date_range(split_date, split_date + datetime.timedelta(days=1), freq='15min')
        # future_df['ds'] = date_range
        future_df['ds'] = date_range
        # future_df['y'] = pd.NA
        future_df['unique_id'] = 'power'
        
        return future_df[1:]
        
    def create_dataset(self, ts_data):
        ts_data = self.format_data(ts_data)
        
        split_date = ts_data['ds'].iloc[-(96 * 1 + 1)] if MODE == 'test' else ts_data['ds'].iloc[-1]
    
        train_set = ts_data[ts_data['ds'] <= split_date]
        test_set = ts_data[ts_data['ds'] > split_date]
        
        future_df = self.create_future_df(split_date)
        return train_set, test_set, future_df
    
    def train_pred(self, train, test, future_df):
        
        # 训练
        self.nf.fit(df=train)
        
        # 输出
        prediction = self.nf.predict(futr_df=future_df).rename(columns={f'{MODEL}':'y'})
        
        return prediction
            
###############################################################################
# 随机种子
SEED = 2026

# 预测时长
PRED_POINTS = 96

# 输入时长
INPUT_POINTS = 96 * 7

# 模型
MODEL = 'NHITS' # 'TSMixerx', 'NHITS'

# 训练轮数
EPOCH = 500

# batch
BATCH_SIZE = 512

# 损失函数
LOSS = 'MSE' # 'MAE', 'MSE'

# LR
LR = 1e-2

# 标准化方法
NORMAL_METHOD = 'minmax' # 'standard', 'minmax', 'robust'

# 模式，测试模式取最后一天为测试集
MODE = 'test'

###############################################################################

if __name__=='__main__':
    start_time = time.time()
    
    logger = init_logger()
    
    set_random_seed()
    
    ts_data = read_ts_data()
    
    # figure(ts_data, '2024/1/1 0:00:00', '2024/1/10 0:00:00')
    
    forecaster = Forecaster()
    
    train, test, future_df = forecaster.create_dataset(ts_data)
    
    pred = forecaster.train_pred(train, test, future_df)
    
    figure_pred(train, test, pred)
    
    RMSE = root_mean_squared_error(test['y'], pred['y'])
    result = 1/(1+RMSE)
    print(f'结果={result}')
    
    time_cost = time_conver(time.time()-start_time)
    logger.info(f'\nmodel: {MODEL}\nepoch: {EPOCH}\nbatch_size: {BATCH_SIZE}\nloss func: {LOSS}\nlr: {LR}\nmormal method: {NORMAL_METHOD}\ntime cost: {time_cost}\nscore: {result}\n')
    
    pass


























