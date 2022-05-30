
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
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.models import *


import wandb
wandb.init()
from utils.args import parse_args
args = parse_args(jupyter=False)
print(args)
import numpy as np
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device is {device}")


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


def get_input_data(data_list, config):
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

data_dir = args.data_dir
lbe_dict = pickle.load(open(os.path.join(data_dir,'lbe_dict.pkl'),'rb'))
if args.cv==-1:
    data_path = os.path.join(data_dir,'data_list.pkl')
else:
    data_path = os.path.join(data_dir,f'data_list_{args.cv}.pkl')
    
data_list = pickle.load(open(data_path,'rb'))
train_data, dev_data, test_data = data_list

remove_features = [x for x in args.remove_features.split(",") if len(x)!=0]

# sparse_features = ['SubjectID', 'ProblemID', 'AssignmentID', 'same_success_num', 'same_fail_num', 'same_avg_attempt', 'success_num', 'fail_num', 'avg_attempt']
# sparse_features = ['SubjectID', 'ProblemID', 'AssignmentID', 'same_success_num', 'same_fail_num', 'same_avg_attempt']
sparse_features = ['SubjectID', 'ProblemID', 'AssignmentID']
sparse_features = [x for x in sparse_features if x not in remove_features]


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
len(field_info_map)
# dense_features = ["s_rate","same_s_rate"]
# dense_features = ["same_s_rate"]
dense_features = []

dense_features = [x for x in dense_features if x not in remove_features]

config = {"dense_features": dense_features,
          "sparse_features": sparse_features, 'sparse_emb_dim': args.sparse_emb_dim}

print("start train use config:{}".format(config))
# 1.get data
dnn_feature_columns, linear_feature_columns, train_model_input, dev_model_input, test_model_input = get_input_data(
    data_list, config)

print(f"dnn_feature_columns is {dnn_feature_columns}")
output_dir = args.output
os.makedirs(output_dir,exist_ok=True)
target_col = "Label"

# 2.early_stop
filepath = os.path.join(output_dir, f'{str(uuid.uuid4())}.h5')
early_stopping_monitor = EarlyStopping(
    monitor="val_binary_crossentropy", patience=args.patience)
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_crossentropy',
                             verbose=1, save_best_only=True, mode='auto', period=1, save_weights_only=True)

adt_config = {"adt_type":args.adt_type,
            "adt_alpha":args.adt_alpha,
            "adt_epsilon":args.adt_epsilon,
            "adt_k":args.adt_k}
# 3.define model
model = eval(args.model_name)(linear_feature_columns, dnn_feature_columns,
                    task='binary',device=device,seed=args.seed,adt_config=adt_config,dnn_dropout=args.dnn_dropout)
model.compile(Adam(model.parameters(), args.lr),
              'binary_crossentropy', metrics=['binary_crossentropy'])


# 4.training
history = model.fit(train_model_input, train_data[target_col].values,
                    batch_size=args.batch_size, epochs=200, verbose=2,
                    validation_data=(
                        dev_model_input, dev_data[target_col].values),
                    callbacks=[early_stopping_monitor, checkpoint])
# 5.load best model
model.load_state_dict(torch.load(filepath))

# 6.predict the test datasets
dev_data['pred'] = model.predict(dev_model_input, batch_size=args.batch_size)
model_report = get_model_metrics(dev_data['Label'], dev_data['pred'])
test_data['pred'] = model.predict(test_model_input, batch_size=args.batch_size)
model_report['filepath'] = filepath
model_report['dev_pred'] = ",".join([str(x) for x in dev_data['pred'].tolist()])
model_report['test_pred'] = ",".join([str(x) for x in test_data['pred'].tolist()])
wandb.log(model_report)
