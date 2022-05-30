import os
import pandas as pd
from datetime import datetime
def change2timestamp(t):
    timeStamp = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def load_raw_data(data_dir):
    early = pd.read_csv(os.path.join(data_dir,'early.csv'))
    late = pd.read_csv(os.path.join(data_dir,'late.csv')) 
    main_table = pd.read_csv(os.path.join(data_dir,'Data/MainTable.csv'))
    main_table['ms_timestamp'] = main_table['ServerTimestamp'].apply(change2timestamp)
    code_state = pd.read_csv(os.path.join(data_dir,'Data/CodeStates/CodeStates.csv'))
    subject = pd.read_csv(os.path.join(data_dir,'Data/LinkTables/Subject.csv'))
    metadata = pd.read_csv(os.path.join(data_dir,'Data/DatasetMetadata.csv'))
    return early,late,main_table,code_state,subject,metadata