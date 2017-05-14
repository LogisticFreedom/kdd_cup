# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from scipy.interpolate import lagrange
import numpy as np
import copy
import matplotlib.pyplot as plt

def read_weather(input_file,output_file):
    # 每天为单位保存数值
    date_vec=[]
    value = [[] for i in xrange(0,7)]
    start_time = 0
    flag = 1
    myfile = open(input_file)
    file_len = len(myfile.readlines())
    myfile.close()
    with open(input_file,'r') as f:
        f.readline()  # ignore the title
        cnt = 0
        for num,line in enumerate(f):
            sample = line.split(',')
            if cnt < 8:
                time_stamp = datetime.strptime(sample[0], '%Y/%m/%d')
                time_stamp = time_stamp + timedelta(hours=int(sample[1]))   # 获取每个节点的时间
                # print time_stamp
                for i in xrange(0,7):
                    value[i].append(float(sample[i+2]))

                date_vec.append(time_stamp)
                for i in xrange(0,8):  #插入20分钟时间
                    time_stamp = time_stamp+timedelta(minutes=20)
                    date_vec.append(time_stamp)
                cnt +=1
            elif(cnt >= 8 or num == file_len-1):  # 对一天的数据进行插值处理
                # 检查有无异常值
                for i in xrange(0,7):
                    tmp = copy.deepcopy(value[i])
                    tmp = sorted(tmp)
                    for j in xrange(len(tmp)):
                        mid = tmp[len(tmp)/2]
                        if(abs(value[i][j])>abs(mid)*10):
                            if(j>0 and j<len(tmp)-1):
                                value[i][j] = (value[i][j-1] + value[i][j+1]) /2
                            elif(j == 0):
                                value[i][j] = value[i][j+1]
                            else:
                                value[i][j] = value[i][j-1]
                x = np.array([1])
                if(cnt >= 8):
                    x = np.array([float(i) for i in xrange(0,9)])  # 9个点，8个区间
                    #第二天0点需要考虑，作为插值的参考
                    for i in xrange(0,7):
                        value[i].append(float(sample[i+2]))
                elif(num == file_len-1):
                    x = np.array([float(i) for i in xrange(0, 8)])  # 8个点，7个区间
                for index,v in enumerate(value):  # 遍历不同的weather值
                    position = 1 # 插入值所在的位置
                    y = np.array(value[index])  #处理每个维度的插值
                    for i in xrange(0,8):  #每三个小时插值8个点
                        # 调用拉格朗日插值，得到插值函数p
                        p = lagrange(x, y)
                        xx = np.linspace(i, i+1, 8)
                        yy = p(xx)   #插值得到的值

                        value[index][position:position] = yy.tolist()
                        position = position+len(yy.tolist())+1
                        # 数值若小于0 则赋值为0
                        for j in xrange(len(value[index])):
                            if value[index][j] < 0:
                                value[index][j] = 0
                #去除多余的第二天的点的值
                if (cnt >= 9):
                    for i in xrange(0,7):
                        value[i].pop()
                #输出结果
                with open(output_file,'a') as o:
                    if(flag==1):
                        title = "date,time,pressure,sea_pressure,wind_direction,wind_speed,temperature,rel_humidity,precipitation\n"
                        o.write(title)
                        flag=0
                    line = ""
                    for index in xrange(len(date_vec)):
                        line += datetime.strftime(date_vec[index],"'%Y-%m-%d','%H:%M:%S'")+","
                        for j in xrange(len(value)):
                            line += str(round(value[j][index],3))+","
                        o.write(line+"\n")
                        line = ""

                #先清空，然后放入第二天的值
                date_vec = []
                value = [[] for i in xrange(0, 7)]
                time_stamp = datetime.strptime(sample[0], '%Y/%m/%d')
                time_stamp = time_stamp + timedelta(hours=int(sample[1]))  # 获取每个节点的时间
                # print time_stamp
                for i in xrange(0, 7):
                    value[i].append(float(sample[i + 2]))

                date_vec.append(time_stamp)
                for i in xrange(0, 8):  # 插入20分钟时间
                    time_stamp = time_stamp + timedelta(minutes=20)
                    date_vec.append(time_stamp)
                cnt = 1

if __name__ == '__main__':
    read_weather("/Users/admin/Desktop/kdd/dataSets/testing_phase1/weather (table 7)_test1.csv","weather_test.csv")
    read_weather("/Users/admin/Desktop/kdd/dataSets/training/weather (table 7)_training.csv","weather_train.csv")