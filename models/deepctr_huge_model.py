import torch
torch.set_num_threads(2) 
import torch
import torch.nn as nn
import os
import pandas as pd
from utils.metrics import get_model_metrics
from utils.data_loader import *
import pickle
from models.loss import *
from .adt_utils import *
import random
from utils.utils import set_seed
import uuid
import torch
from torch.optim import Adagrad,Adam
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.models import *
from deepctr_torch.models.basemodel import BaseModel
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tensorflow.keras.utils import to_categorical

class DeepctrHuge(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, args):
        super().__init__(linear_feature_columns, dnn_feature_columns, device=args.device)

        self.xdeepfm = xDeepFM(linear_feature_columns, dnn_feature_columns,
                               task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.deepfm = DeepFM(linear_feature_columns, dnn_feature_columns,
                             task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.onn = ONN(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.autoint = AutoInt(linear_feature_columns, dnn_feature_columns,
                               task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.wdl = WDL(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.nfm = NFM(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.afn = AFN(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.difm = DIFM(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout,att_head_num=2)
        self.ifm = IFM(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.fibi = FiBiNET(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.dcn = DCN(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)
        self.dncm = DCNMix(linear_feature_columns, dnn_feature_columns,
                       task='binary', device=args.device, seed=args.seed, dnn_dropout=args.dnn_dropout)


        num_model = 6*2
        self.args = args
        self.save_path = args.save_path
        self.loss_type = args.loss_type
        self.raw_x_dim = self.compute_input_dim(dnn_feature_columns)
        self.merge_model_raw = nn.Sequential(nn.Linear(num_model*2+self.raw_x_dim+2, num_model, bias=True),
                                             nn.ReLU(),
                                             nn.Linear(num_model, num_model, bias=True))
        self.out_layer = nn.Sequential(nn.Linear(num_model*5+2, 2, bias=True),)
        self.drop_num = args.drop_num
        self.dropouts = nn.ModuleList(
            [nn.Dropout(min(0.9, round(0.1*i+0.1, 1))) for i in range(self.drop_num)])
        
        self.optim = Adam(self.parameters(), args.lr)
        
        
        #对抗
        self.adt_config = args.adt_config
        self.adt_type = self.adt_config.get("adt_type","None")
        self.adt_alpha = self.adt_config.get("adt_alpha",0.2)
        self.adt_epsilon = self.adt_config.get("adt_epsilon",1)
        self.adt_k = self.adt_config.get("adt_k",3)

    def loss(self, y_pred, y_true,reduction="mean"):
        if self.loss_type == "ce":
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
        elif self.loss_type == "focal":
            loss_fct = FocalLoss(gamma=self.args.focal_gamma,alpha=self.args.focal_alpha, reduction=reduction)
        # elif self.loss_type == 'poly':
        #     loss_fct = PolyLoss(epsilon=self.args.poly_eepsilon, reduction="mean")
        # elif self.loss_type == "dice":
        #     # config find in https://github.com/ShannonAI/dice_loss_for_NLP/blob/master/scripts/ner_enconll03/bert_dice.sh
        #     loss_fct = DiceLoss(with_logits=True, smooth=1, ohem_ratio=0,
        #                         alpha=0.01, square_denominator=True,
        #                         index_label_position=True, reduction="mean")
        #     y_pred = y_pred.to(torch.float32)
        else:
            raise ImportError(f"The loss func is error {self.loss_type}")
        loss = loss_fct(y_pred, y_true)
        return loss

    def forward(self, X, labels=None):
        #raw feature
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        dnn_input = combined_dnn_input(
            sparse_embedding_list, dense_value_list)
        # dnn_output = self.dnn(dnn_input)
        # print(f"dnn_input shape is {dnn_input.shape}")
       
    
        
        #merge model result
        output_xdeepfm = self.xdeepfm(X)
        output_deepfm = self.deepfm(X)
        output_onn = self.onn(X)
        output_autoint = self.autoint(X)
        output_wdl = self.wdl(X)
        output_nfm = self.nfm(X)
        output_afn = self.afn(X)
        output_difm = self.difm(X)
        output_ifm = self.ifm(X)
        output_fibi = self.fibi(X)
        output_dcn = self.dcn(X)
        output_dncm = self.dncm(X)
        

        x_model = torch.cat([output_xdeepfm, output_deepfm, output_onn,
                             output_autoint, output_wdl, output_nfm,output_afn,output_difm,output_ifm,output_fibi,output_dcn,output_dncm
                             ], axis=-1)
        x_model_gate = torch.sigmoid(x_model)*x_model
        x_model_avg = torch.mean(x_model,axis=-1).unsqueeze(-1)
        x_model_std = torch.std(x_model,axis=-1).unsqueeze(-1)
        # print(x_model_avg.shape)
        #merge raw feature
        x_merge = torch.cat([x_model_gate,x_model_avg,x_model_std,x_model,dnn_input], axis=-1)
        
        # print(f"x_merge shape is {x_merge.shape}")
        x_merge = self.merge_model_raw(x_merge)
        # print(f"x_merge shape is {x_merge.shape}")
        
        # merge result
        x_merge = torch.cat([x_model_avg,x_model_std,
                             x_merge,
                             x_model,
                             x_merge+x_model,
                             torch.sigmoid(x_merge)*x_model,
                             x_merge*torch.sigmoid(x_model)
                             ], axis=-1)
        # print(f"x_merge shape is {x_merge.shape}")
        logits = 0
        loss = 0
        for _, dropout in zip(range(self.drop_num), self.dropouts):
            one_logits = self.out_layer(dropout(x_merge))
            # print(f"one_logits shape is {one_logits.shape}")
            # print(f"one_logits is {one_logits}")
            logits += one_logits/self.drop_num
            if labels is not None:
                one_loss = self.loss(one_logits, labels)
                loss += one_loss/self.drop_num
        logits = torch.softmax(logits, dim=-1) 
        return logits, loss

    def predict(self, x, batch_size=256):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        x = self.format_input(x)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x)[0].cpu().data.numpy()
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans).astype("float64")

    def evaluate(self, x, y, batch_size=256):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        y_pred = self.predict(x, batch_size)[:, 1]
        eval_result = get_model_metrics(y_pred=y_pred, y_true=y)
        return y_pred,eval_result
    
    def format_input(self,x):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        return x
                
    def save_model(self):
        torch.save(self.state_dict(), self.save_path)
        
    def load_model(self):
        self.load_state_dict(torch.load(self.save_path))
        
    def fit(self, x=None, y=None,dev_x=None, dev_y=None,test_x=None,test_y=None,batch_size=None, epochs=1, verbose=1,shuffle=True,patience=5):
        x = self.format_input(x)
        train_tensor_data = Data.TensorDataset(
                            torch.from_numpy(
                                np.concatenate(x, axis=-1)),
                            torch.from_numpy(to_categorical(y)))
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)
        optim = self.optim
        
        # 对抗学习
        if self.adt_type=='fgm':
            fgm = FGM(self)
            print("Training use FGM ~~")
        elif self.adt_type == 'pgd':
            pgd = PGD(self)
            print("Training use PGD ~~")
        elif self.adt_type == 'freeat':
            freeat = FreeAT(self)
            print("Training use FreeAT ~~")
        elif self.adt_type == 'freelb':
            freelb = FreeLB(self)
            print("Training use FreeLB ~~")
        else:
            print("Training use none adt ~~")
        
        
        best_auc = -1
        patience_count = 0
        for epoch in range(epochs):
            print(f"Start epoch {epoch}")
            loss_epoch = 0
            self.train()
            for x_train, y_train in train_loader:
                x_train = x_train.to(self.device).float()
                y_train = y_train.to(self.device).float()
                y_pred,loss = self(x_train,y_train)
                optim.zero_grad()
                loss_epoch+=loss.item()
                loss.backward()
                
                if self.adt_type=='fgm':#使用fgm对抗训练方式
                    #对抗训练
                    fgm.attack(self.adt_epsilon, "embedding") # 在embedding上添加对抗扰动
                    y_pred,loss_adv = self(x_train,y_train)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore("embedding") # 恢复embedding参数
                    #梯度下降更新参数                
                optim.step()
            _,train_eval_result = self.evaluate(x,y, batch_size)
            # print(f"train_eval_result is {train_eval_result['AUC_Value']}")
            # eval
            y_dev_pred,y_test_pred = None,None
            y_dev_pred,eval_result = self.evaluate(dev_x,dev_y, batch_size)
            if eval_result['AUC_Value']>best_auc:
                print(f"dev auc is improve from {best_auc} to {eval_result['AUC_Value']}")
                best_auc = eval_result['AUC_Value']
                self.save_model()
                patience_count = 0
            else:
                print(f"dev auc not improve from {best_auc}, now auc is {eval_result['AUC_Value']}")
                if patience_count>patience:
                    self.load_model()
                    y_dev_pred = self.predict(dev_x)[:,1]
                    y_test_pred = self.predict(test_x)[:,1]
                    print(f"Early stop, best auc is {best_auc}")
                    break
                patience_count+=1
            print(f"train loss_epoch is {round(loss_epoch,4)},train auc is {train_eval_result['AUC_Value']},val auc is {eval_result['AUC_Value']}")
        return y_dev_pred,y_test_pred