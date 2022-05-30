import os
import pandas as pd
from utils.data_loader import *
import numpy as np
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error

train_data_dir = "data/csedm_2021/datashop/F19_Release_Train_06-28-21/Train/"
early,late,main_table,code_state,subject,metadata = load_raw_data(train_data_dir)


test_data_dir = "data/csedm_2021/datashop/F19_Release_Test_06-28-21/Test"
early_test,late_test,main_table_test,code_state_test,subject_test,metadata_test = load_raw_data(test_data_dir)


# Split data

test_stu_ids = early['SubjectID'].unique().tolist()
test_stu_ids = test_stu_ids[:100]


train_early = early[~early['SubjectID'].isin(test_stu_ids)]
train_late = late[~late['SubjectID'].isin(test_stu_ids)]

test_early = early[early['SubjectID'].isin(test_stu_ids)]
test_late = late[late['SubjectID'].isin(test_stu_ids)]


# Get features
# 
# Use the feature from first 30 questions to predict the final score.


import xgboost
from sklearn.model_selection import GridSearchCV



subject_2_score = dict(zip(subject['SubjectID'],subject['X-Grade']))


problem_ids = [1,   3,   5,  12,  13,  17,  20,  21,  22,  24,  25,  28,  31,
               32,  33,  34,  36,  37,  38,  39,  40, 100, 101, 102, 128, 232,
               233, 234, 235, 236]

# 1 correct，0 error，0.5 no interaction
def get_feature(df, main_table, subject_2_score, mode='train'):
    x_list = []
    y_list = []
    stu_ids = []
    for stu_id, group in tqdm_notebook(df.groupby("SubjectID")):
        stu_ids.append(stu_id)
        x_stu = []
        stu_all_main_tabel = main_table[main_table["SubjectID"] == stu_id]
        label_list = []
        attempts_list = []
        correct_list = []
        duration_list = []
        max_score_list = []
        for problem in problem_ids:
            problem_group = group[group["ProblemID"] == problem]
            stu_main_tabel = stu_all_main_tabel[stu_all_main_tabel["ProblemID"] == problem]
            if len(problem_group) != 0:
                label = int(problem_group.iloc[0]["Label"])
                attempts = int(problem_group.iloc[0]["Attempts"])
                correct = int(problem_group.iloc[0]["CorrectEventually"])
                duration = (stu_main_tabel['ms_timestamp'].max(
                )-stu_main_tabel['ms_timestamp'].min())//1000/60
                max_score = stu_main_tabel['Score'].fillna(0).max()

                label_list.append(label)
                attempts_list.append(attempts)
                correct_list.append(correct)
                duration_list.append(duration)
                max_score_list.append(max_score)
            else:
                if len(stu_main_tabel) != 0:
                    print(stu_id)
                label = 0.5
                attempts = 0
                correct = 0
                duration = 0
                max_score = 0
            x_stu.extend([label, attempts, correct, duration, max_score])
        x_list.append(x_stu)
        if mode == 'train':
            y_list.append(subject_2_score[stu_id])
    x = np.array(x_list)
    if mode == 'train':
        y = np.array(y_list)
        return x, y, stu_ids
    return x, stu_ids


x_train,y_train,_ = get_feature(train_early,main_table,subject_2_score)
x_test,y_test,_ = get_feature(test_early,main_table,subject_2_score)

x_final_test,stu_ids = get_feature(early_test,main_table_test,subject_2_score={},mode="test")


# Training

#  Grid search 
# You can change params' range as you want
# param_grid = {
#     'random_state': [4, 2022, 449, 3028, 12, 14],
#     'max_depth': [1, 2, 3, 4, 5, 6, 7],
#     'n_estimators': range(0, 100, 1),
#     'subsample': np.arange(0, 1, 0.01),
# }
# quick run
param_grid = {
    # 'random_state': [15],
    'max_depth': [1],
    'n_estimators': [15],
    'subsample': [0.5],
}

model = xgboost.XGBRegressor(tree_method='gpu_hist', gpu_id=2, n_jobs=20)

clf = GridSearchCV(model, param_grid, cv=5,
                   scoring='neg_mean_squared_error', verbose=0)
clf.fit(x_train, y_train)
print(f"cv score is {clf.best_score_}")
# final model

best_params = clf.best_params_

model = xgboost.XGBRegressor(
    tree_method='gpu_hist', gpu_id=3, n_jobs=10, **best_params)
model.fit(x_train, y_train)

# evalute
y_test_pred = model.predict(x_test)
test_mse = mean_squared_error(y_test,y_test_pred)
print(f"Test mse is {test_mse}")

# submit
x_final_test,stu_ids = get_feature(early_test,main_table_test,subject_2_score={},mode="test")
y_final_pred = model.predict(x_final_test)


df_submit = pd.DataFrame({"SubjectID":stu_ids,"X-Grade":y_final_pred})
df_submit.to_csv('data/submit/track2/predictions.csv',index=False)
