import os
import pandas as pd
from utils.data_loader import *
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import pickle
from utils.task2_multi_config import *

import lightgbm as lgb
print('LGBM Version', lgb.__version__)
import catboost
from catboost import CatBoostRegressor
print('catboost Version', catboost.__version__)
import xgboost
print('xgboost Version', xgboost.__version__)

from sklearn.ensemble import GradientBoostingRegressor

def load_pickle(path):
    return pickle.load(open(path,'rb'))


def train_cat_1fold(x_train, y_train, x_dev, y_dev, x_test):
    model = CatBoostRegressor(**CAT_PARAMS)
    model.fit(
        x_train, y_train,
        eval_set=(x_dev, y_dev),
        use_best_model=True,
        verbose=100
    )
    y_dev_pred = model.predict(x_dev)
    y_test_pred = model.predict(x_test)
    return y_dev_pred,y_test_pred


def train_xgb_1fold(x_train, y_train, x_dev, y_dev, x_test):
    
    model = xgboost.XGBRegressor(tree_method='gpu_hist',gpu_id=3, n_jobs=10,**XGB_PARAMS)
    
    model.fit(
        x_train, y_train,
        verbose=100
    )
    y_dev_pred = model.predict(x_dev)
    y_test_pred = model.predict(x_test)
    return y_dev_pred,y_test_pred


def train_lgbm_1fold(x_train, y_train, x_dev, y_dev, x_test):
    
    d_train = lgb.Dataset(x_train, label=y_train)
    d_val = lgb.Dataset(x_dev, label=y_dev, reference=d_train)
    # d_test = lgb.Dataset(x_test, reference=d_train)
    model = lgb.train(
        train_set=d_train,
        valid_sets=[d_train, d_val],
        valid_names=['train', 'valid'],
        params=LGB_MDOEL_PARAMS,
        **LGB_TRAIN_PARAMS
    )
    y_dev_pred = model.predict(x_dev)
    y_test_pred = model.predict(x_test)
    return y_dev_pred,y_test_pred

def train_gbdt_1fold(x_train, y_train, x_dev, y_dev, x_test):
    model = GradientBoostingRegressor(**XGB_PARAMS)
    model.fit(
        x_train, y_train,
    )
    y_dev_pred = model.predict(x_dev)
    y_test_pred = model.predict(x_test)
    return y_dev_pred,y_test_pred

from sklearn.ensemble import AdaBoostRegressor
def train_ada_1fold(x_train, y_train, x_dev, y_dev, x_test):
    model = AdaBoostRegressor(**ADA_PARAMS)
    model.fit(
        x_train, y_train,
    )
    y_dev_pred = model.predict(x_dev)
    y_test_pred = model.predict(x_test)
    return y_dev_pred,y_test_pred

def get_fold_data(fold,x_all,y_all,split_info,feature_index):
    """获取多折的数据并进行特征筛选
    """
    x_train = x_all[split_info[fold][2]][:,feature_index]
    y_train = y_all[split_info[fold][2]]
    x_dev = x_all[split_info[fold][3]][:,feature_index]
    y_dev = y_all[split_info[fold][3]]
    return x_train,y_train,x_dev,y_dev


def get_model_feature(pred_dict,model_list,mode='train'):
    """融合多折模型预测的结果"""
    model_feature_list = []
    for model in model_list:
        if mode=='train':
            model_feature = np.hstack(pred_dict[model])
        else:
            model_feature = np.mean(pred_dict[model],axis=0)
        model_feature_list.append(model_feature)
    x_model_feature = np.array(model_feature_list).T
    return x_model_feature

def train_step1(x_all, y_all,x_test,split_info, feature_index,model_list):
    """第一阶段训练

    Args:
        x_all (_type_): 所有的训练特征，不需要划分多折
        y_all (_type_): 所有训练标签
        x_test (_type_): 最后提交的
        split_info (_type_): 多折划分的结果
        feature_index (_type_): 选取的特征列表
        model_list (_type_): 训练使用的模型
    Returns:
        _type_: _description_
    """
    dev_pred_dict = {}
    test_pred_dict = {}
    x_test = x_test[:,feature_index]
    for fold in tqdm_notebook(range(5)):
        x_train, y_train, x_dev, y_dev = get_fold_data(
            fold, x_all, y_all, split_info, feature_index=feature_index)
        for model in model_list:
            if model not in dev_pred_dict:
                dev_pred_dict[model] = []
                test_pred_dict[model] = []
            y_dev_pred, y_test_pred = eval(
                f"train_{model}_1fold(x_train, y_train, x_dev, y_dev, x_test)")
            dev_pred_dict[model].append(y_dev_pred)
            test_pred_dict[model].append(y_test_pred)
    x_meta_train = get_model_feature(dev_pred_dict, model_list,mode='train')
    x_meta_test = get_model_feature(test_pred_dict, model_list,mode='test')
    return x_meta_train, x_meta_test

def add_agg_meta(x):
    """给原始特征增加统计信息

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    add_x = np.vstack([np.max(x, axis=-1), np.min(x, axis=-1),
     np.std(x, axis=-1), np.mean(x, axis=-1)]).T
    return np.hstack([x,add_x])

def merge_stage1_features(x_model_list,x_raw,feature_index):
    x_model = add_agg_meta(np.hstack(x_model_list))#模型预测的特征
    x_final = np.hstack([x_model,x_raw[:,feature_index]])#拼接原始特征
    return x_final