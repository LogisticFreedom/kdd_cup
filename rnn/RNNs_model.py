import numpy as np
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN
from keras.models import Sequential

from rnn import train_rnn


class RNNsModel(object):

    # 初始化RNN模型参数，包括输入维度、隐藏层维度、输出维度、cell单元类型
    def __init__(self, inputDim, hiddenNum, outputDim, unit):

        self.inputDim = inputDim
        self.hiddenNum = hiddenNum
        self.outputDim = outputDim
        self.buildModel(unit)

    # 建立RNN模型
    def buildModel(self,unit = "GRU"):

        self.model = Sequential()
        if unit == "GRU":
            self.model.add(GRU(self.hiddenNum, input_shape=(None, self.inputDim)))
        elif unit == "LSTM":
            self.model.add(LSTM(self.hiddenNum, input_shape=(None, self.inputDim)))
        elif unit == "RNN":
            self.model.add(SimpleRNN(self.hiddenNum, input_shape=(None, self.inputDim)))
        #self.model.add(Dense(self.hiddenNum//2))
        self.model.add(Dense(self.outputDim))
        self.model.compile(loss="mean_absolute_percentage_error", optimizer='rmsprop', metrics=["mean_absolute_percentage_error"])

    # 正常训练，可以设置训练轮数、batch大小
    def train(self,trainX,trainY,epoch,batchSize):
        self.model.fit(trainX, trainY, epochs=epoch, batch_size=batchSize, verbose=1, validation_split=0.0)

    # 预测
    def predict(self,testX):
        pred = self.model.predict(testX)
        return pred.reshape(-1)

    # 迭代预测
    def forcastingMultiAhead(self, testX, aheadNum):

        buffer = testX
        ans = []
        for i in range(aheadNum):
            testX = train_rnn.createTestSamples(buffer, lookBack=30)
            res = self.model.predict(testX)
            buffer = np.delete(buffer, 0)
            buffer = np.append(buffer, res)
            ans.append(res[0,0])
        return np.array(ans)
