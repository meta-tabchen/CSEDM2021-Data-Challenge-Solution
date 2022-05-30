# Solution of CSEDM Data Challenge 2021
You can download the data from [here](https://sites.google.com/ncsu.edu/csedm-dc-2021/home).

## Create conda env

1.Use the following command to create a new conda enviroment.

`conda env create -f conda_env.yml`

2.Install the modified Deepctr-torch[1] package

```
cd other_projects/DeepCTR-Torch
pip install -e .
```

To improve the generalization of models, we added several adversarial training methods to this package.


## Track1 inference

See `track1 inference.ipynb`, you need to modify the `test_data_dir`  we use three models to predict 

The `test_data_dir` folder's structure is follwing:

```
├── Data
│   ├── CodeStates
│   │   └── CodeStates.csv
│   ├── DatasetMetadata.csv
│   ├── LinkTables
│   │   └── Subject.csv
│   └── MainTable.csv
├── early.csv
└── late.csv
```
The score maybe close to **0.7965**.

## Track2 inference

See `track2 inference.ipynb`, you need to modify the `test_data_dir`, they use the same folder in `Track1 inference`.

The score maybe close to **161**.

## Track1 Training

In this task, we try to use methods from the recommender system to predict future students' performance. The first is extracting features, see the jupyter notebook in `track1_extract_features.ipynb`. Then, we build two types of models to predict the test datasets. Finally, we tried different strategies to merge different models' results.

- single model: we try almost all models in DeepCTR.
- multi huge model: we merge multi models in DeepCTR into one huge model and add multi-dropout and adversarial training to improve the performance in test datasets.

### Train single model
You can train by the command `python deepctr.py`


### Train multi huge model
You can train by the command `python deepctr_huge.py`

### Configs
We use the [wandb](https://wandb.ai/) to manage the results of each model. All config can be found in `wandb_configs`

You can start one sweep by the following command (one group params search)

`wandb sweep wandb_configs/deepctr_5cv.yaml -p csedm_2021`



## Track2 Training
In this trace, we extracted many features from the first 30 questions' interactions to predict each student's final grade. To be specific, first, we build two types of models to model each student's performance. Second, we try to use the different strategies to merge the result from each model.

- one stage: see `track2_one_stage.py`
- multi stage: see `track2_multi_stage.ipynb`  


## References
1. [deepctr-torch](https://github.com/shenweichen/DeepCTR-Torch)
2. [CSEDM2021 official website](https://sites.google.com/ncsu.edu/csedm-dc-2021/home) 
3. https://github.com/haradai1262/NeurIPS-Education-Challenge-2020


## Contact
Email: tabchen2808581543@gmail.com (Jiahao Chen)