import torch
torch.set_num_threads(2) 
import os
from utils.metrics import get_model_metrics
from utils.data_loader import *
from utils.utils import set_seed
import uuid
from sklearn.metrics import *
from models.deepctr_huge_model import DeepctrHuge
from utils.deepctr_utils import load_data,get_input_data
from utils.huge_args import parse_args
import wandb
wandb.init()
args = parse_args(jupyter=False)
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

adt_config = {"adt_type": args.adt_type,
              "adt_alpha": args.adt_alpha,
              "adt_epsilon": args.adt_epsilon,
              "adt_k": args.adt_k}
args.adt_config = adt_config

output_dir = args.output
os.makedirs(output_dir,exist_ok=True)
args.save_path = os.path.join(output_dir, f'{str(uuid.uuid4())}.h5')
print(f"args is {args}")


data_list,config,lbe_dict = load_data(args)
train_data, dev_data, test_data = data_list
print("start train use config:{}".format(config))
# 1.get data
dnn_feature_columns, linear_feature_columns, train_model_input, dev_model_input, test_model_input = get_input_data(
    data_list, config,lbe_dict)
args.dnn_feature_columns = dnn_feature_columns
args.linear_feature_columns = linear_feature_columns

huge_model = DeepctrHuge(linear_feature_columns=linear_feature_columns,
                         dnn_feature_columns=dnn_feature_columns,args=args)
huge_model = huge_model.to(device)


y_dev_pred,y_test_pred = huge_model.fit(x=train_model_input, y=train_data['Label'],
               dev_x=dev_model_input, dev_y=dev_data['Label'], test_x=test_model_input,
               batch_size=256, epochs=args.epochs, verbose=1, shuffle=True, patience=args.patience)

model_report = get_model_metrics(dev_data['Label'], y_dev_pred)
dev_data['pred'] = y_dev_pred
test_data['pred'] = y_test_pred
model_report['filepath'] = args.save_path 
model_report['dev_pred'] = ",".join([str(x) for x in dev_data['pred'].tolist()])
model_report['test_pred'] = ",".join([str(x) for x in test_data['pred'].tolist()])
# model_report['args'] = vars(args)
wandb.log(model_report)
