import argparse
def parse_args(jupyter=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,default="xDeepFM", required=False)
    parser.add_argument("--lr", type=float,default=1e-2,required=False)
    parser.add_argument("--data_dir", type=str, default="data/csedm_2021/processed/", required=False)
    parser.add_argument("--output", type=str, default="model_save/deepctr", required=False)
    parser.add_argument("--batch_size", type=int, default=512, required=False)
    parser.add_argument("--sparse_emb_dim", type=int, default=128, required=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cv", type=int, default=-1)
    parser.add_argument("--adt_type", type=str, default="None")
    parser.add_argument("--adt_alpha", type=float, default=0.2)
    parser.add_argument("--adt_epsilon", type=float, default=1)
    parser.add_argument("--adt_k", type=int, default=3)
    parser.add_argument("--dnn_dropout", type=float, default=0)
    parser.add_argument("--remove_features", type=str, default="")
    
    
    if jupyter:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args
