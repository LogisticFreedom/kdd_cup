import xgboost as xgb
import pandas as pd
import  numpy as np


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

        dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
        dtest = xgb.DMatrix(X_test, missing=np.nan)
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.005
        params["min_child_weight"] = 6
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.9
        params["scale_pos_weight"] = 1
        params["silent"] = 1
        params["max_depth"] = 10

        plst = list(params.items())

        num_rounds = 800

        rf = xgb.train(plst, dtrain, num_rounds)

        pred = rf.predict(dtest)
        pred = np.rint(pred)
        acc = SMAPE(y_test, pred)
        scores.append(acc)
    print("score mean:", np.mean(scores))
    print("score std:", np.std(scores))
    return scores

def xgbTrain(features):

    dataDF = pd.read_csv("./data/train12.csv")
    dataDF = dataDF[dataDF["volume"] != 0]  # 丢弃ytrue为0的情况，不仅在交叉验证中，实际训练也丢弃了
    trainX = dataDF[features].values
    trainY = dataDF['volume'].values
    print("trainx shape", trainX.shape)
    print("trainY shape", trainY.shape)

    scores = crossValidation(trainX, trainY, cvRate=0.96, cvEpoch=20)
    print(scores)

if __name__ == "__main__":
    features = ['tollgate_id',
               'direction',
       'day_of_week', 'is_weekend',
       #'month',
        # 'day',
        'hour', 'minute', 'time_win_label',
                'lengthSum',
       'widthSum', 'lanesSum', 'laneWidthSum',
                'intersectionNum',
                'linkNum',
       'max', 'mean', 'mean_1h', 'mean_20', 'mean_40', 'median', 'min',
       'range', 'std',
                # 'pressure', 'sea_pressure', 'wind_direction', 'wind_speed',
                # 'temperature', 'rel_humidity', 'precipitation'
                ]
    xgbTrain(features)




