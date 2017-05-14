import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

# 0-(1,0） 1-(1,1) 2-(2,0) 3-(3,0) 4-(3,1)
def getID(x):
    if x == 0 or x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3 or x == 4:
        return 3

def getDirection(x):
    if x == 0 or x == 2 or x == 3:
        return 0
    if x == 1 or x==4:
        return 1


def paraseFeature():

    train = pd.read_csv("features_train(with head).csv")
    test = pd.read_csv("features_test(with head).csv")

    train["tollgate_id"] = train["gate_dir"].apply(getID)
    train["direction"] = train["gate_dir"].apply(getDirection)
    train["time"] = train["time"].apply(lambda x: datetime.strptime(x, "'%H:%M:%S'").time())
    train.drop("gate_dir", axis=1, inplace=True)

    test["tollgate_id"] = test["gate_dir"].apply(getID)
    test["direction"] = test["gate_dir"].apply(getDirection)
    test.drop("gate_dir", axis=1, inplace=True)
    test.drop("total_volume", axis=1, inplace=True)

    test["time"] = test["time"].apply(lambda x: datetime.strptime(x, "'%H:%M:%S'"))
    test["time"] = test["time"].apply(lambda x: (x+timedelta(hours=2)).time())

    train.to_csv("volume_feature_train.csv", index=False)
    test.to_csv("volume_feature_test.csv", index=False)

if __name__ == "__main__":
    paraseFeature()





