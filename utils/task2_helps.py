import numpy as np

problem_ids = [1,   3,   5,  12,  13,  17,  20,  21,  22,  24,  25,  28,  31,
               32,  33,  34,  36,  37,  38,  39,  40, 100, 101, 102, 128, 232,
               233, 234, 235, 236]

def get_feature(df, main_table, subject_2_score, mode='train'):
    x_list = []
    y_list = []
    stu_ids = []
    for stu_id, group in df.groupby("SubjectID"):
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

