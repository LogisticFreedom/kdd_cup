import pandas as pd
from datetime import datetime, timedelta
from math import *
import sklearn.preprocessing as preprocessing


def creat_feature():
    volume = pd.read_csv("../data/training_20min_avg_volume_original.csv")
    volume["start_time"] = volume["time_window"].apply(
        lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))
    start_time_tmp = datetime(2016, 9, 20, 0, 0, 0)
    start_time_list = []
    # 72 time window per day; 29 days in trainset
    for i in range(72 * 28):
        start_time_list.append(start_time_tmp + timedelta(minutes=20 * i))

    # fill
    tollgate = [1, 2, 3]
    direction = [0, 1]
    for i in tollgate:
        for j in direction:
            df_list = []
            for k in start_time_list:
                if (i != 2 or j != 1) and volume[volume.start_time == k][volume.tollgate_id == i][
                            volume.direction == j].empty:
                    time_window_tmp = "[" + k.strftime("%Y-%m-%d %H:%M:%S") + "," + (
                        k + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S") + ")"
                    df_list.append([i, time_window_tmp, j, 0, k])
            df = pd.DataFrame(df_list, columns=volume.columns)
            volume = volume.append(df).reset_index(drop=True)
    scaler = preprocessing.StandardScaler()
    volume["date"] = volume["start_time"].apply(lambda x: datetime.date(x))
    volume["time"] = volume["start_time"].apply(lambda x: datetime.time(x))
    start_time_tmp = datetime(2016, 9, 20, 0, 0, 0)
    start_time_list = []
    # 72 time window per day; 29 days in trainset
    for i in range(72 * 28):
        start_time_list.append(start_time_tmp + timedelta(minutes=20 * i))
    list = [1, 3, 5, 7, 15, 30, 50, 72]
    s_pre_df = pd.DataFrame()
    for l in list:
        pre_list1 = []
        pre_list2 = []
        pre_list3 = []
        pre_list4 = []
        pre_list5 = []
        for i in tollgate:
            for j in direction:
                for k in start_time_list:
                    if (i != 2 or j != 1):
                        pre_list1.append(i)
                        pre_list2.append(j)
                        pre_list3.append(k)
                        pre_list4.append((k - timedelta(minutes=l * 20)).date())
                        pre_list5.append((k - timedelta(minutes=l * 20)).time())
        pre_df = pd.DataFrame()
        pre_df["tollgate_id"] = pre_list1
        pre_df["direction"] = pre_list2
        pre_df["datetime"] = pre_list3
        pre_df["date"] = pre_list4
        pre_df["time"] = pre_list5
        if (l == 1):
            s_pre_df = pre_df.copy()
        else:
            s_pre_df = pd.merge(s_pre_df, pre_df, on=["tollgate_id", "direction", "datetime"])
        s_pre_df = pd.merge(s_pre_df, volume, on=["tollgate_id", "direction", "date", "time"], how="left")
        s_pre_df.rename(columns=lambda x: x.replace("volume", "feature_" + str(l)), inplace=True)
        del s_pre_df["date"]
        del s_pre_df["start_time"]
        del s_pre_df["time"]
        del s_pre_df["time_window"]
    s_pre_df["date"] = s_pre_df["datetime"].apply(lambda x: datetime.date(x))
    s_pre_df["time"] = s_pre_df["datetime"].apply(lambda x: datetime.time(x))
    pre_feature = s_pre_df.copy()
    drop_date_start = datetime.strptime('2016-09-30', '%Y-%m-%d').date()
    for i in range(10):
        pre_feature = pre_feature[pre_feature["date"] != drop_date_start + timedelta(hours=(i + 1) * 24)]
    pre_feature.to_csv("window_feature.csv", index=False)


creat_feature()
