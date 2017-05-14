import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def volumeFeatureExtractTrain():

    # 读取数据，转换时间格式
    data = pd.read_csv("../data/training_20min_avg_volume_original.csv")
    data["start_time"] = data["time_window"].apply(lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))


    volumeFeature = pd.DataFrame()

    # 按id和方向分组
    group = data.groupby(["tollgate_id", "direction"])
    for key, item in group:
        print(key)
        item = item.sort_values(by="start_time") # 按时间排序
        item.index = pd.to_datetime(item["start_time"]) # 设置时间索引，方便resample
        item = item.resample('2H')["volume"] # 两小时采样

        # 在每个两个小时里做统计
        for c, subItem in item:
            mean = subItem.mean()
            mean_20 = subItem.iloc[5:6].mean()
            mean_40 = subItem.iloc[4:6].mean()
            mean_1h = subItem.iloc[3:6].mean()
            median = subItem.median()
            minNum = subItem.min()
            maxNum = subItem.max()
            rangeNum = maxNum-minNum
            std = subItem.std()
            #difVol = subItem.diff().dropna()
            print([mean, mean_20, mean_40, mean_1h, median, minNum, maxNum, rangeNum, std])
            #print(difVol.values)
            c = c+timedelta(hours=2)
            print(c)

            # 设置返回数据格式
            for i in range(6):
                volumeRecord = pd.Series()
                volumeRecord["tollgate_id"] = key[0]
                volumeRecord["direction"] = key[1]
                volumeRecord["date"] = c.date()
                volumeRecord["time"] = c.time()
                volumeRecord["satrt_time"] = str(c)
                c = c+timedelta(minutes=20)
                volumeRecord["mean"] = mean
                volumeRecord["mean_20"] = mean_20
                volumeRecord["mean_40"] = mean_40
                volumeRecord["mean_1h"] = mean_1h
                volumeRecord["median"] = median
                volumeRecord["min"] = minNum
                volumeRecord["max"] = maxNum
                volumeRecord["range"] = rangeNum
                volumeRecord["std"] = std
                volumeFeature = volumeFeature.append(volumeRecord, ignore_index=True)
    return volumeFeature

def volumeFeatureExtactTest():

    # 读取数据，转换时间格式
    data = pd.read_csv('../data/test1_20min_avg_volume_original.csv')
    data["start_time"] = data["time_window"].apply(lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))

    volumeFeature = pd.DataFrame()

    # 按id和方向分组
    group = data.groupby(["tollgate_id", "direction"])
    for key, item in group:
        print(key)
        item = item.sort_values(by="start_time") # 按时间排序，重要

        # 由于这里下午的时间不是每两个小时连续的，所以直接硬编码，用循环处理
        for i in range(0, 12 * 7, 6):

            c = item.iloc[i]["start_time"]
            print(c)
            c = c + timedelta(hours=2)

            subItem = item.iloc[i:i+6]["volume"]
            mean = subItem.mean()
            mean_20 = subItem.iloc[5:6].mean()
            mean_40 = subItem.iloc[4:6].mean()
            mean_1h = subItem.iloc[3:6].mean()
            median = subItem.median()
            minNum = subItem.min()
            maxNum = subItem.max()
            rangeNum = maxNum - minNum
            std = subItem.std()
            for i in range(6):
                volumeRecord = pd.Series()
                volumeRecord["tollgate_id"] = key[0]
                volumeRecord["direction"] = key[1]
                volumeRecord["date"] = c.date()
                volumeRecord["time"] = c.time()
                volumeRecord["satrt_time"] = str(c)
                c = c+timedelta(minutes=20)
                volumeRecord["mean"] = mean
                volumeRecord["mean_20"] = mean_20
                volumeRecord["mean_40"] = mean_40
                volumeRecord["mean_1h"] = mean_1h
                volumeRecord["median"] = median
                volumeRecord["min"] = minNum
                volumeRecord["max"] = maxNum
                volumeRecord["range"] = rangeNum
                volumeRecord["std"] = std
                volumeFeature = volumeFeature.append(volumeRecord, ignore_index=True)
    return volumeFeature




if __name__ == "__main__":

    volumeFeature = volumeFeatureExtractTrain()
    print(volumeFeature)
    volumeFeature.to_csv("volume_feature_2h_train_nofill.csv", index=False)
    volumeFeature = volumeFeatureExtactTest()
    print(volumeFeature)
    volumeFeature.to_csv("volume_feature_2h_test_nofill.csv", index=False)





