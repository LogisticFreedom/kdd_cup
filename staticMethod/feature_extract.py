import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from sklearn.preprocessing import LabelEncoder

def featureExtract():

    # 读取原始数据文件
    data = pd.read_csv("../data/training_20min_avg_volume_original.csv")
    idFeature = pd.read_csv("tollgate_feature.csv")
    weatherFeature = pd.read_csv("new_weather_train.csv")

    data["start_time"] = data["time_window"].apply(lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))
    #testData["start_time"] = testData["time_window"].apply(
     #   lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))

    # 时间窗口完整列表生成
    start_time_tmp = datetime(2016, 9, 19, 0, 0, 0)
    start_time_list = []
    # 72 time window per day; 29 days in trainset
    for i in range(72 * 29):
        start_time_list.append(start_time_tmp + timedelta(minutes=20 * i))

    #用0填补空值，变为10440个完整数据
    tollgate = [1, 2, 3]
    direction = [0, 1]
    for i in tollgate:
        for j in direction:
            df_list = []
            for k in start_time_list:
                if (i != 2 or j != 1) and data[data.start_time == k][data.tollgate_id == i][data.direction == j].empty:
                    time_window_tmp = "[" + k.strftime("%Y-%m-%d %H:%M:%S") + "," + (
                    k + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S") + ")"
                    df_list.append([i, time_window_tmp, j, 0, k])
            df = pd.DataFrame(df_list, columns=data.columns)
            data = data.append(df).reset_index(drop=True)

    #data = pd.concat([data, testData], axis=0)

    # 提取时间特征
    data["date"] = data["start_time"].apply(lambda x: datetime.date(x))
    data["time"] = data["start_time"].apply(lambda x: datetime.time(x))
    data["day_of_week"] = data["start_time"].apply(lambda x: x.isoweekday())

    # 提取早晚高峰标示
    data["morning_peak"] = data["time"].apply(lambda x: 1 if x >= time(hour=8,minute=0,second=0) and x <= time(hour=10,minute=0,second=0) else 0)
    data["evening_peak"] = data["time"].apply(lambda x: 1 if x >= time(hour=17,minute=0,second=0) and x <= time(hour=19,minute=0,second=0) else 0)

    # 星期几one-hot
    weekDayOnehot = pd.get_dummies(data["day_of_week"], prefix="week_day")
    data = pd.concat([data, weekDayOnehot], axis=1)

    # 分解时间特征
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 6 else 0)
    data["month"] = data["date"].apply(lambda x: x.month)
    data["day"] = data["date"].apply(lambda x: x.day)
    data["hour"] = data["time"].apply(lambda x: x.hour)
    data["minute"] = data["time"].apply(lambda x: x.minute)

    # 排序
    # data = data.sort(["date", "time"])

    # 时间窗口编码
    enc = LabelEncoder()
    enc.fit(data["time"])
    data["time_win_label"] = enc.transform(data["time"])

    # 去除9.30-10.8的9天的数据
    drop_date_start = datetime.strptime('2016-09-30', '%Y-%m-%d').date()
    data_drop = data.copy()
    for i in range(9):
        data_drop = data_drop[data_drop["date"] != drop_date_start + timedelta(hours=i * 24)]

    # 加入tollgate-id的特征
    data_drop = pd.merge(data_drop, idFeature, how="left")
    # weatherFeature["time"] = weatherFeature["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))
    # data_drop = pd.merge(data_drop, weatherFeature, on="time", how="left")
    #data_drop = pd.merge(data_drop, trainVolumeFeature, how="outer", on=["tollgate_id", "time", "date", "direction"])
    return data_drop, enc

def featureExtractTest(enc):

    data = pd.read_csv("../data/submission_sample_volume.csv")
    idFeature = pd.read_csv("tollgate_feature.csv")
    #testVolumeFeature = pd.read_csv("volume_feature_test.csv")
    weatherFeature = pd.read_csv("new_weather_test.csv")

    data["start_time"] = data["time_window"].apply(lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))
    # 提取时间特征
    data["date"] = data["start_time"].apply(lambda x: datetime.date(x))
    data["time"] = data["start_time"].apply(lambda x: datetime.time(x))
    data["day_of_week"] = data["start_time"].apply(lambda x: x.isoweekday())

    data["morning_peak"] = data["time"].apply(
        lambda x: 1 if x >= time(hour=8, minute=0, second=0) and x <= time(hour=10, minute=0, second=0) else 0)
    data["evening_peak"] = data["time"].apply(
        lambda x: 1 if x >= time(hour=17, minute=0, second=0) and x <= time(hour=19, minute=0, second=0) else 0)
    # 星期几one-hot
    weekDayOnehot = pd.get_dummies(data["day_of_week"], prefix="week_day")
    data = pd.concat([data, weekDayOnehot], axis=1)
    # 分解时间特征
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 6 else 0)
    data["month"] = data["date"].apply(lambda x: x.month)
    data["day"] = data["date"].apply(lambda x: x.day)
    data["hour"] = data["time"].apply(lambda x: x.hour)
    data["minute"] = data["time"].apply(lambda x: x.minute)
    # 编码时间窗口，必须用上面的enc
    data["time_win_label"] = enc.transform(data["time"])

    data = pd.merge(data, idFeature, how="left")
    # weatherFeature["time"] = weatherFeature["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))
    # data = pd.merge(data, weatherFeature, on="time", how="left")
    #data = pd.merge(data, testVolumeFeature, how="outer", on=["tollgate_id", "time", "date", "direction"])

    return data

if __name__ == "__main__":

    traindata, enc = featureExtract()
    print (traindata.columns)
    traindata.to_csv("train7.csv", index=False)
    testdata = featureExtractTest(enc)
    print(testdata.columns)
    testdata.to_csv("test7.csv", index=False)