import pandas as pd
import  numpy as np
from datetime import datetime, timedelta

def featureCombination_Weather(originalFileName, newFileName, outputFileName):

    feature = pd.read_csv(originalFileName)
    weatherFeature = pd.read_csv(newFileName)

    weatherFeature["start_time"] = weatherFeature["start_time"].apply(lambda x: datetime.strptime(x, "%Y/%m/%d %H:%M"))
    weatherFeature["date"] = weatherFeature["start_time"].apply(lambda x: datetime.date(x))
    weatherFeature["time"] = weatherFeature["start_time"].apply(lambda x: datetime.time(x))

    feature["start_time"] = feature["start_time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    feature["date"] = feature["start_time"].apply(lambda x: datetime.date(x))
    feature["time"] = feature["start_time"].apply(lambda x: datetime.time(x))


    newFeature = pd.merge(feature, weatherFeature)

    newFeature.to_csv(outputFileName, index=False)

def featureCombination_volume(originalFileName, newFileName, outputFileName):

    feature = pd.read_csv(originalFileName)
    volumeFeature = pd.read_csv(newFileName)

    volumeFeature["date"] = volumeFeature["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    volumeFeature["time"] = volumeFeature["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))


    feature["date"] = feature["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    feature["time"] = feature["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))


    newFeature = pd.merge(feature, volumeFeature)

    newFeature["time"] = newFeature["time"].apply(lambda x: x.time())
    newFeature["date"] = newFeature["date"].apply(lambda x: x.date())

    #newFeature = newFeature.dropna(axis=0)

    newFeature.to_csv(outputFileName, index=False)

def featureCombination_volume_weather(volumeFileName, weatherFileName, outputFileName):

    volume = pd.read_csv(volumeFileName)
    weather = pd.read_csv(weatherFileName)

    volume["date"] = volume["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    volume["time"] = volume["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))

    weather["date"] = weather["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    weather["time"] = weather["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))

    newFeature = pd.merge(volume, weather)

    newFeature["time"] = newFeature["time"].apply(lambda x: x.time())
    newFeature["date"] = newFeature["date"].apply(lambda x: x.date())

    newFeature.to_csv(outputFileName, index=False)


if __name__ == "__main__":

    # featureCombination_Weather("train9.csv", "new_weather_train.csv", "train10.csv")
    # featureCombination_Weather("test9.csv", "new_weather_test.csv", "test10.csv")

    featureCombination_volume("./data/train2.csv", "./data/volume_feature_2h_train_nofill.csv", "train12.csv")
    featureCombination_volume("./data/test2.csv", "./data/volume_feature_2h_test_nofill.csv", "test12.csv")

    # featureCombination_volume_weather("train11.csv", "train9.csv", "train8.csv")
    # featureCombination_volume_weather("test11.csv", "test7.csv", "test8.csv")