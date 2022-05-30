
import torch
torch.set_num_threads(2) 
import os
import pandas as pd
from utils.metrics import get_model_metrics
from utils.data_loader import *
import pickle

import random
from utils.utils import set_seed
import uuid
import torch
from torch.optim import Adagrad,Adam
from deepctr_torch.inputs import SparseFeat, DenseFeat,VarLenSparseFeat, get_feature_names

key2index = {'For': 1,
             'ArrayIndex': 2,
             'If/Else': 3,
             'Math+-*/': 4,
             'LogicAndNotOr': 5,
             'LogicCompareNum': 6,
             'NestedIf': 7,
             'LogicBoolean': 8,
             'StringFormat': 9,
             'DefFunction': 10,
             'StringConcat': 11,
             'StringIndex': 12,
             'StringLen': 13,
             'Math%': 14,
             'While': 15,
             'StringEqual': 16,
             'CharEqual': 17,
             'NestedFor': 18}

field_info_map = {'SubjectID': "user",
                  'ProblemID': "item",
                  'AssignmentID': "concept",
                  "concept": "concept",
                  "same_success_num": "same",
                  "same_fail_num": "same",
                  "same_s_rate": "same",
                  "same_avg_attempt": "same",
                  "success_num": "all",
                  "fail_num": "all",
                  "s_rate": "all",
                  "avg_attempt": "all"
                  }

def get_input_data(data_list, config,lbe_dict):
    train_data, dev_data, test_data = data_list
    # 获取配置信息
    dense_features = config['dense_features']
    sparse_features = config['sparse_features']
    sparse_emb_dim = config['sparse_emb_dim']
    dense_emb_dim = 1

    feature_names = sparse_features+dense_features
    field_info = [field_info_map[x] for x in feature_names]
    # 添加场信息
    fixlen_feature_columns = [
        SparseFeat(feat,
                   vocabulary_size=len(lbe_dict[feat].classes_),
                   embedding_dim=sparse_emb_dim,
                   group_name=field_info[i])
        for i, feat in enumerate(sparse_features)
    ] + [DenseFeat(
        feat,
        dense_emb_dim,
    ) for feat in dense_features]
    have_concept = False
    if have_concept:
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('concepts', vocabulary_size=len(
        key2index) + 1, embedding_dim=sparse_emb_dim), maxlen=8, combiner='mean')]
        dnn_feature_columns = fixlen_feature_columns+varlen_feature_columns
        linear_feature_columns = fixlen_feature_columns+varlen_feature_columns
    else:
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

    # 构建输入
    train_model_input = {name: train_data[name+"_encoded"] for name in feature_names}
    dev_model_input = {name: dev_data[name+"_encoded"] for name in feature_names}
    test_model_input = {name: test_data[name+"_encoded"] for name in feature_names}
    if have_concept:
        train_model_input['concepts'] = np.vstack(train_data['concepts'])
        dev_model_input['concepts'] = np.vstack(dev_data['concepts'])
        test_model_input['concepts'] = np.vstack(test_data['concepts'])
    
    # 对dense单独处理
    for graph_emb_feature in dense_features:
        for data in [train_model_input, dev_model_input, test_model_input]:
            data[graph_emb_feature] = np.vstack(data[graph_emb_feature])
    return dnn_feature_columns, linear_feature_columns, train_model_input, dev_model_input, test_model_input



def load_data(args):
    data_dir = args.data_dir
    lbe_dict = pickle.load(open(os.path.join(data_dir,'lbe_dict.pkl'),'rb'))
    if args.cv==-1:
        data_path = os.path.join(data_dir,'data_list.pkl')
    else:
        data_path = os.path.join(data_dir,f'data_list_{args.cv}.pkl')
        
    data_list = pickle.load(open(data_path,'rb'))
    train_data, dev_data, test_data = data_list

    remove_features = [x for x in args.remove_features.split(",") if len(x)!=0]
    sparse_features = ['SubjectID', 'ProblemID', 'AssignmentID']
    sparse_features = [x for x in sparse_features if x not in remove_features]


    len(field_info_map)
    # dense_features = ["same_s_rate"]
    # dense_features = [x for x in dense_features if x not in remove_features]
    dense_features = []

    config = {"dense_features": dense_features,
            "sparse_features": sparse_features, 'sparse_emb_dim': args.sparse_emb_dim}

    return data_list,config,lbe_dict