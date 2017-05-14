import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle


# 计算SMAPE，主要是应对ytrue是0的情况
def SMAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 2.0

# 计算MAPE，需要提前丢弃ytrue是0的情况
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 交叉验证
def crossValidation(trainX, trainY, cvRate = 0.90, cvEpoch = 20):

    scores = []
    for i in range(cvEpoch):
        X, Y = shuffle(trainX, trainY) # 打乱数据
        offset = int(X.shape[0] * cvRate)
        X_train, y_train = X[:offset], Y[:offset]
        X_test, y_test = X[offset:], Y[offset:]
        #y_train = np.log(y_train+1)
        #rf = SVR(epsilon=0.01, C = 0.1)
        #rf = RandomForestRegressor(n_estimators=100)
        rf = GradientBoostingRegressor(
                                    loss='ls'
                                    , learning_rate=0.05,
                                    n_estimators=100,
                                     subsample=1.0,
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

        # 四舍五入成整数
        #pred = np.exp(pred)
        pred = np.rint(pred)
        acc = SMAPE(y_test, pred)
        scores.append(acc)

    # 去除评分中的inf
    scores = [x for x in scores if str(x) != 'nan' and str(x) != 'inf']
    print("score mean:", np.mean(scores))
    print("score std:", np.std(scores))
    return scores


train = pd.read_csv("train5.csv")
test = pd.read_csv("test5.csv")

features = [   'is_weekend',
                   #'month',
                 'day',
                'hour', 'minute',
                'time_win_label',
                'morning_peak',
                'evening_peak',
       #          'lengthSum',
       # 'widthSum', 'lanesSum',
       #          'laneWidthSum',
       #          'intersectionNum',
               ]

groupTrain = train.groupby(["tollgate_id", "direction"])
groupTest = test.groupby(["tollgate_id", "direction"])
groups = {}

for key, item in groupTrain:
    groups[key] = [item]
for key, item in groupTest:
    groups[key].append(item)

result = pd.DataFrame()

for i in range(5):

    keys = sorted(groups.keys())
    key = keys[i]
    item = groups[key]
    print("tollgate id:", key[0])
    print("direction:", key[1])

    trainx = item[0][features]
    trainy = item[0]["volume"]
    testx = item[1][features]
    print("trainx shape:", trainx.values.shape)
    print("trainy shape:", trainy.values.shape)
    print("testx shape:", testx.values.shape)

    score = crossValidation(trainx,trainy)
    print("score:", score)

    rf = GradientBoostingRegressor(
        loss='ls'
        , learning_rate=0.1,
        n_estimators=100
        , subsample=1.0,
        # , min_samples_split=2
        # , min_samples_leaf=1
        max_depth=6,
        # , init=None
        random_state=0,
        # , max_features=None
        # , alpha=0.9
        # , verbose=0
        # , max_leaf_nodes=None
        # , warm_start=False
    )

    rf.fit(trainx, trainy)
    print("feature importance", rf.feature_importances_)
    ans = np.rint(rf.predict(testx))
    #print(ans)
    resultFeatures = ['tollgate_id', 'time_window', 'direction', 'volume']
    resultDF = item[1][resultFeatures]
    resultDF["volume"] = ans
    result = result.append(resultDF)

result.to_csv("divide_result_gbdt.csv", index=False)

