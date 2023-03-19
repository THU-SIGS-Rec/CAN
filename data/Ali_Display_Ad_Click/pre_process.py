import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    behavior_data = data_frame=pd.read_csv('./raw_data/raw_sample.csv')
    user_data = data_frame=pd.read_csv('./raw_data/user_profile.csv').sample(frac=0.1)
    ad_data = data_frame=pd.read_csv('./raw_data/ad_feature.csv').sample(frac=0.1)
    print("User number", user_data.shape)
    print("Ad number", ad_data.shape)
    
    
    user_data.set_index('userid')
    ad_data.set_index('adgroup_id')
    behavior_data = pd.merge(behavior_data, user_data, left_on='user', right_on='userid')
    behavior_data = pd.merge(behavior_data, ad_data, left_on='adgroup_id', right_on='adgroup_id')
    behavior_data = behavior_data.dropna(axis=0, how='any')
    
    train_df = behavior_data[behavior_data['time_stamp'] < datetime(2017, 5, 12).timestamp()]
    vallid_df = behavior_data[(behavior_data['time_stamp'] > datetime(2017, 5, 12).timestamp()) &
                              (behavior_data['time_stamp'] < datetime(2017, 5, 13).timestamp())]
    test_df = behavior_data[behavior_data['time_stamp'] > datetime(2017, 5, 13).timestamp()]
    
    train_df.to_csv('train.csv')
    vallid_df.to_csv('valid.csv')
    test_df.to_csv('test.csv')
    
    print(train_df.shape)
    print(vallid_df.shape)
    print(test_df.shape)
    