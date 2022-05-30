import os
import json
import pickle
import torch
import pandas as pd
from utils.utils import set_seed
from utils.data_loader import *
from torch.optim import Adam
from deepctr_torch.inputs import SparseFeat, DenseFeat,VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *



class ARGS:
    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)


def load_model_config(model_id,model_dir):
    model_path = f'{model_dir}/{model_id}_model.h5'
    params = json.load(open(f'{model_dir}/{model_id}_config.json'))
    params['model_path'] = model_path
    args = ARGS(params)
    return args


field_info_map = {'SubjectID': "user",
                  'ProblemID': "item",
                  'AssignmentID': "concept",
                  }


def get_input_data(test_data, config,lbe_dict):
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

    test_model_input = {
        name: test_data[name+"_encoded"] for name in feature_names}
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    return dnn_feature_columns, linear_feature_columns, test_model_input