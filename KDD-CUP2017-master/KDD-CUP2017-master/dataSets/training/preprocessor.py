#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 19/4/2017 上午9:49
@Author  : Icy Huang
@Site    : 
@File    : preprocessor.py
@Software: PyCharm Community Edition
@Python  : 2.7.10

python preprocessor.py volume_training.csv --t 20 --output volume_training_parsed.csv
python preprocessor.py volume_test1.csv --t 20 --output volume_test_parsed.csv
"""
# KDD 交通流量预测 数据预处理

from datetime import datetime, timedelta
from business_calendar import Calendar, MO, TU, WE, TH, FR
import click


@click.command()
@click.argument('input_file')  # , help='the csv file you wanna process'
@click.option('--t', default=20, help='time interval')
@click.option('--output', default='volume_parsed.csv', help='the name of output file')
def preprocessor(input_file, t, output):
    # （tollgate_id,direction）五个方向 0-(1,0） 1-(1,1) 2-(2,0) 3-(3,0) 4-(3,1)
    # 三维特征  vehicle_mode{0-7}, has_etc{0,1}, vehicle_type{0,1,unknown}
    corr_pair = [[1, 0], [1, 1], [2, 0], [3, 0], [3, 1]]
    time_point = dict()  # 三个维度的各类别
    volume_per_slot = dict()  # 车流量
    working_day = dict()  # 是否工作日
    sample_cnt = 0
    time_interval = t
    output_file = output
    time_slot_cnt = 0
    with open(input_file, 'r') as f:
        f.readline()  # ignore the title
        # 输入前先将数据集递增排序
        for line in f:
            sample = line.split(',')
            time_stamp = datetime.strptime(sample[0], '%d/%m/%Y %H:%M')
            if sample_cnt == 0:
                mins = time_stamp.minute % 10
                if mins == 0:
                    begin_time = time_stamp
                else:
                    begin_time = time_stamp - timedelta(minutes=mins)
            minutes = (time_stamp - begin_time).seconds / 60
            if minutes >= time_interval or sample_cnt == 0:
                # 每隔time_interval 初始化一次
                mins = time_stamp.minute % 10
                if mins == 0:
                    begin_time = time_stamp
                else:
                    begin_time = time_stamp - timedelta(minutes=mins)
                index = datetime.strftime(begin_time, '%d/%m/%Y %H:%M')
                volume_per_slot[index] = [0 for _ in range(5)]
                time_point[index] = [dict() for _ in range(5)]
                for i in range(5):
                    time_point[index][i]['mode'] = [0 for _ in range(8)]
                    time_point[index][i]['etc'] = [0 for _ in range(2)]
                    time_point[index][i]['type'] = [0 for _ in range(3)]
                time_slot_cnt += 1
            for i in range(5):
                if str(corr_pair[i][0]) in sample[1] and str(corr_pair[i][1]) in sample[2]:
                    volume_per_slot[index][i] += 1
                    time_point[index][i]['mode'][int(sample[3])] += 1
                    time_point[index][i]['etc'][int(sample[4])] += 1
                    if '1' in sample[5] or '0' in sample[5]:
                        time_point[index][i]['type'][int(sample[5])] += 1
                    else:
                        time_point[index][i]['type'][2] += 1
                    break
            sample_cnt += 1

    with open(str(time_interval)+u'分钟'+output_file, 'w') as f:
        title_str = '"time_window","gate_dir",'
        for i in range(8):
            title_str += 'vehicle_model[' + str(i) + '],'
        for i in range(2):
            title_str += 'has_etc[' + str(i) + '],'
        for i in range(2):
            title_str += 'vehicle_type[' + str(i) + '],'
        title_str += 'no_type,total_volume,working_day'
        f.write(title_str + '\n')
        cal = Calendar()
        for k in volume_per_slot.keys():
            for j in range(5):
                time1 = datetime.strptime(k, '%d/%m/%Y %H:%M')
                time_formalize1 = datetime.strftime(time1, "%Y-%m-%d %H:%M:%S")
                time2 = time1 + timedelta(minutes=time_interval)
                time_formalize2 = datetime.strftime(time2, "%Y-%m-%d %H:%M:%S")
                f.write('"[%s,%s)",' % (time_formalize1, time_formalize2))
                # gate_dir
                f.write('"%d",' % j)
                for m in range(8):
                    f.write('"%d",' % time_point[k][j]['mode'][m])
                for e in range(2):
                    f.write('"%d",' % time_point[k][j]['etc'][e])
                for t in range(2):
                    f.write('"%d",' % time_point[k][j]['type'][t])
                f.write('"%d",' % time_point[k][j]['type'][2])
                f.write('"%d",' % volume_per_slot[k][j])
                if cal.isbusday(time1):
                    working_day[k] = 1
                else:
                    working_day[k] = 0
                f.write('"%d"\n' % working_day[k])

if __name__ == '__main__':
    preprocessor()