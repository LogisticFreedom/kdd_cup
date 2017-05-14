import datetime

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from rnn import RNNs_model as RNNs


# 获取时间序列，返回df形式
def getTimeSeries(filename):

    volume = pd.read_csv(filename)

    id1 = volume[volume["tollgate_id"] == 1]
    id1out = id1[id1["direction"] == 1]["volume"]
    id1in = id1[id1["direction"] == 0]["volume"]
    id2 = volume[volume["tollgate_id"] == 2]
    id2in = id2[id2["direction"] == 0]["volume"]
    id3 = volume[volume["tollgate_id"] == 3]
    id3out = id3[id3["direction"] == 1]["volume"]
    id3in = id3[id3["direction"] == 0]["volume"]

    return id1out, id1in, id2in, id3out, id3in

# 分割时间序列作为样本，lookBack为窗口大小,并整理成RNN的输入形式
def createSamples(dataset, lookBack, RNN = True):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        a = dataset[i:(i + lookBack)]
        dataX.append(a)
        dataY.append(dataset[i + lookBack])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
    return dataX,dataY

# 测试数据整理为RNN输入形式 并进行补齐
def createTestSamples(testts, lookBack):
    testts = testts.reshape((1,-1))

    dataX = np.reshape(testts, (testts.shape[0], testts.shape[1], 1))

    dataX = pad_sequences(dataX, maxlen=lookBack, dtype='float32')  # 左端补齐

    return dataX

# 划分训练集和测试集
def divideTrainTest(dataset,rate = 0.75):

    train_size = int(len(dataset) * rate)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:]
    return train,test

# RNN训练
def train(ts, lookBack):
    rnn = RNNs.RNNsModel(inputDim = 1, hiddenNum = 100, outputDim = 1, unit = "GRU")
    trainX, trainY = createSamples(ts, lookBack=lookBack)
    rnn.train(trainX, trainY, epoch=60, batchSize=20)
    return rnn

# 多步迭代向前预测
def forecasting(rnn, testts ,aheadNum = 6):
    #testts = createTestSamples(testts, lookBack=30)
    ans = rnn.forcastingMultiAhead(testts, aheadNum)
    #ans = rnn.predict(testts)
    return ans


def createSubmitTable():
    id = [1, 2, 3]
    direction = [1, 0]
    start_time_window1 = datetime.datetime(2016, 10, 18, 8, 0, 0)
    start_time_window2 = datetime.datetime(2016, 10, 18, 17, 0, 0)
    fw = open("rnn_result.csv", 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    for k in id:
        for dir in direction:
            if k == 2 and dir == 1:
                continue
            for i in range(7):
                start_time_window = start_time_window1 + datetime.timedelta(days=1 * i)
                for j in range(6):
                    time_window_end = start_time_window + datetime.timedelta(minutes=20)
                    out_line = ','.join(['"' + str(k) + '"',
                                         '"[' + str(start_time_window) + ',' + str(time_window_end) + ')"',
                                         '"' + str(dir) + '"',
                                         ]) + '\n'
                    start_time_window = time_window_end
                    fw.writelines(out_line)
                for j in range(6):
                    start_time_window = start_time_window2 + datetime.timedelta(days=1 * i)
                    time_window_end = start_time_window + datetime.timedelta(minutes=20)
                    out_line = ','.join(['"' + str(k) + '"',
                                         '"[' + str(start_time_window) + ',' + str(time_window_end) + ')"',
                                         '"' + str(dir) + '"',
                                         ]) + '\n'
                    start_time_window = time_window_end
                    fw.writelines(out_line)
    fw.close()

def getResult():

    id1outTest, id1inTest, id2inTest, id3outTest, id3inTest = getTimeSeries("../data/test1_20min_avg_volume_test.csv")
    id1out, id1in, id2in, id3out, id3in = getTimeSeries("../data/drop_national_day_training.csv")
    trainList = [id1out, id1in, id2in, id3out, id3in]
    testList = [id1outTest, id1inTest, id2inTest, id3outTest, id3inTest]
    ans = []

    for i in range(5):
        ts = trainList[i].values
        testTs = testList[i].values
        rnn = train(ts, lookBack=36)
        n = len(testTs)
        for i in range(0, n, 6):
            tmpAns = forecasting(rnn, testTs[i:i+6])
            ans.append(tmpAns)

    result = np.array(ans).flatten()
    submit = pd.read_csv("rnn_result.csv")
    submit["volume"] = result
    submit.to_csv("rnn_result.csv", index=False)
    return np.array(ans).flatten()

if __name__ == "__main__":
    lookBack = 30
    #rnn = train(id1in.values, lookBack=lookBack)
    createSubmitTable()
    ans = getResult()
    #print(ans.shape)
    #
