import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from staticMethod.feature_extract import featureExtract, featureExtractTest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import  SVR
import xgboost as xgb
from datetime import  time, datetime
from sklearn.utils import shuffle

# 计算SMAPE，主要是应对ytrue是0的情况
def SMAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 2.0

# 计算MAPE，需要提前丢弃ytrue是0的情况
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 交叉验证
def crossValidation(trainX, trainY, cvRate = 0.96, cvEpoch = 20):

    scores = []
    for i in range(cvEpoch):
        X, Y = shuffle(trainX, trainY) # 打乱数据
        offset = int(X.shape[0] * cvRate)
        X_train, y_train = X[:offset], Y[:offset]
        X_test, y_test = X[offset:], Y[offset:]
        #rf = SVR(epsilon=0.01, C = 0.1)
        #rf = RandomForestRegressor(n_estimators=100)
        rf = GradientBoostingRegressor(
                                    #loss='ls'
                                    #, learning_rate=0.1,
                                    n_estimators=120,
                                    # , subsample=1.0
                                    # , min_samples_split=2
                                    # , min_samples_leaf=1
                                    max_depth=6
                                    # , init=None
                                    , random_state=0,
                                    # , max_features=None
                                    # , alpha=0.9
                                    # , verbose=0
                                    # , max_leaf_nodes=None
                                    # , warm_start=False
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        pred = np.rint(pred)
        acc = SMAPE(y_test, pred)
        scores.append(acc)
    scores = [x for x in scores if str(x) != 'nan' and str(x) != 'inf']
    print("score mean:", np.mean(scores))
    print("score std:", np.std(scores))
    return scores

def train(features):
    dataDF = pd.read_csv("new_feature.csv")
    print(dataDF.columns)

    dataDF = dataDF[dataDF["volume"] != 0] # 丢弃ytrue为0的情况，不仅在交叉验证中，实际训练也丢弃了
    trainX = dataDF[features]

    trainY = dataDF['volume']
    print("trainx shape", trainX.values.shape)
    print("trainY shape", trainY.values.shape)
    #rf = RandomForestRegressor(n_estimators=100)
    rf = GradientBoostingRegressor(loss='ls'
                                , learning_rate=0.1
                                , n_estimators=120
                                , subsample=1
                                , min_samples_split=2
                                , min_samples_leaf=1
                                , max_depth=6
                                , init=None
                                , random_state=None
                                , max_features=None
                                , alpha=0.9
                                , verbose=0
                                , max_leaf_nodes=None
                                , warm_start=False)
    #scores = cross_val_score(rf, trainX, trainY, scoring="neg_mean_squared_error",cv=10)
    #print(scores)
    scores = crossValidation(trainX, trainY)
    print("cross validation scores:", scores)
    rf.fit(trainX, trainY)
    print("feature score ", pd.DataFrame(rf.feature_importances_))
    return rf

# def predict(rf, features):
#
#     testDataDF = pd.read_csv("features_test(with head).csv")
#
#     testX = testDataDF[features]
#     print("trainx shape", testX.values.shape)
#     ans = rf.predict(testX)
#     testDataDF["volume"] = ans
#     resultFeatures = ['tollgate_id', 'time_window', 'direction', 'volume']
#     resultDF = testDataDF[resultFeatures]
#     resultDF.to_csv("rf_result.csv", index=False)

def preprocessing():
    data = pd.read_csv("train.csv")
    data["time1"] = data["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S"))
    data["time2"] = data["time1"].apply(lambda x: x.hour + x.minute/100)
    data["hour"] = data["time1"].apply(lambda x: x.hour)
    print(data.columns)
    data.drop("time", axis=1, inplace=True)
    data.drop("time1", axis=1, inplace=True)
    data.to_csv("new_feature.csv",index=False)

if __name__ == "__main__":
    preprocessing()
    features = ["tollgate_id", "direction", "day_of_week", "is_weekend", "time2",
       #          'pressure', 'sea_pressure', 'wind_direction', 'wind_speed',
       # 'temperature', 'rel_humidity', 'precipitation'
                ]
    rf = train(features)
    #predict(rf, features)
