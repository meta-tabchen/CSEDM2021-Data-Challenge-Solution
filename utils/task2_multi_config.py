RANDOM_STATE = 0
LGB_MDOEL_PARAMS = {
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "max_depth": 10,
    "num_leaves": 256,
    "colsample_bytree": 0.8,
    "min_child_weight": 0,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": 20,
}
LGB_TRAIN_PARAMS = {
    "num_boost_round": 2000,
    "early_stopping_rounds": 50,
    "verbose_eval": 100,
}

CAT_PARAMS = {
    'depth': 8,
    'learning_rate': 0.1,
    'bagging_temperature': 0.2,
    'od_type': 'Iter',
    'metric_period': 50,
    'iterations': 3000,
    'od_wait': 10,
    'random_seed': RANDOM_STATE,
}

XGB_PARAMS = {'max_depth': 1, 'n_estimators': 15, 'subsample': 0.5,"random_state":RANDOM_STATE}
ADA_PARAMS = {'n_estimators': 90, 'random_state': 0}