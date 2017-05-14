import pandas as pd
import numpy as np
from staticMethod.util import SMAPE, MAPE, crossValidation, gridSearch, kFoldCV
from staticMethod.model import buildTrainModel

def train(features, index):

    dataDF = pd.read_csv("./data/train9.csv")
    print("original dataset columns:", dataDF.columns)

    dataDF = dataDF[dataDF["volume"] != 0] # 丢弃ytrue为0的情况，不仅在交叉验证中，实际训练也丢弃了
    trainX = dataDF[features]
    trainY = dataDF['volume']
    print("trainx shape", trainX.values.shape)
    print("trainY shape", trainY.values.shape)

    rf = buildTrainModel(modelIndex=index)
    #rf = gridSearch(trainX, trainY, modelIndex=modelIndex)
    rf.fit(trainX, trainY)

    #scores, skscores = crossValidation(trainX, trainY, index)
    scores = kFoldCV(trainX, trainY, modelIndex, k=10)
    print("cross validation scores:", scores)
    #print("sklearn cross validation scores:", skscores)


    if index == 1 or index == 2:
        print("feature score ", pd.DataFrame(rf.feature_importances_))
    return rf

def predict(rf, features):

    testDataDF = pd.read_csv("./data/test9.csv")

    testX = testDataDF[features]
    print("testx shape", testX.values.shape)

    ans = rf.predict(testX)
    ans = np.rint(ans)
    #ans = np.exp(ans)

    testDataDF["volume"] = ans
    resultFeatures = ['tollgate_id', 'time_window', 'direction', 'volume']
    resultDF = testDataDF[resultFeatures]
    resultDF.to_csv("./data/rf_result2.csv", index=False)

if __name__ == "__main__":

    # 选择特征
    features = [
        'tollgate_id',
               'direction',
       'day_of_week',
                  'is_weekend',
       #'month',
        #'day',
        'hour', 'minute',
        'time_win_label',
                'lengthSum',
       'widthSum', 'lanesSum', 'laneWidthSum',
                'intersectionNum',
                'linkNum',
       'max', 'mean', 'mean_1h', 'mean_20', 'mean_40', 'median', 'min',
       'range', 'std',
                # 'pressure', 'sea_pressure', 'wind_direction', 'wind_speed',
                # 'temperature', 'rel_humidity', 'precipitation'
                ]

    # 模型序号  # 输入参数为模型序号，1是GBDT，2是随机森林,3是xgboost, 4是adaboost回归，5是多层感知器，6是k近邻回归 7是模型融合stacking
    modelIndex = 3


    rf = train(features, modelIndex)
    #predict(rf, features)
