# encoding = utf-8

import re
import time
import datetime

def get_features(filename, type):
    """
    KDD CUP
    从key_time.csv表格中提取特征

    """

    begin1 = datetime.datetime.strptime("8:00:00", "%H:%M:%S")
    end1 = datetime.datetime.strptime("10:00:00", "%H:%M:%S")
    begin2 = datetime.datetime.strptime("17:00:00", "%H:%M:%S")
    end2 = datetime.datetime.strptime("19:00:00", "%H:%M:%S")
    save = {}

    with open(filename, 'r') as fr, open("features.csv", 'w', encoding='utf-8') as fw:
        fr.readline()
        fw.write("date,time,gate_dir,vehicle_model[0]_2h_ago,vehicle_model[1]_2h_ago,vehicle_model[2]_2h_ago,"
                 "vehicle_model[3]_2h_ago,vehicle_model[4]_2h_ago,vehicle_model[5]_2h_ago,vehicle_model[6]_2h_ago,"
                 "vehicle_model[7]_2h_ago,has_etc[0]_2h_ago,has_etc[1]_2h_ago,working_day_2h_ago,total_volume_2h_ago,"
                 "vehicle_model[0]_1d2h_ago,vehicle_model[1]_1d2h_ago,vehicle_model[2]_1d2h_ago,"
                 "vehicle_model[3]_1d2h_ago,vehicle_model[4]_1d2h_ago,vehicle_model[5]_1d2h_ago,"
                 "vehicle_model[6]_1d2h_ago,vehicle_model[7]_1d2h_ago,has_etc[0]_1d2h_ago,has_etc[1]_1d2h_ago,"
                 "working_day_1d2h_ago,total_volume_1d2h_ago,total_volume\n")
        for line in fr:
            line = re.split(',', line)
            line[-1] = line[-1].strip('\n')
            # 获得gate_dir
            gate_dir = line[2]
            # 获得时间(字符串类型)
            date_time_str = ','.join(line[0:2])
            timer_str = line[1]
            # 获得时间(时间类型)
            date_time = datetime.datetime.strptime(date_time_str, "'%Y-%m-%d','%H:%M:%S'")
            time_ = datetime.datetime.strptime(timer_str, "'%H:%M:%S'")
            # 一行存储的index
            index = ','.join([date_time_str, gate_dir])
            save[index] = line
            # 对于每一个时间点, 都找它的前2个小时 和 前1天2小时

            if type == "train":
                date_time_2h = date_time - datetime.timedelta(hours=2)
                date_time_1d2h = date_time - datetime.timedelta(days=1, hours=2)
            elif type == "test":
                date_time_2h = date_time - datetime.timedelta(hours=0)  # 由于是测试集, 不需要往后推两个小时
                date_time_1d2h = date_time - datetime.timedelta(days=1, hours=0)  # 同上, 只需往后推一天

            date_time_2h_str = date_time_2h.strftime("'%Y-%m-%d','%H:%M:%S'")
            date_time_1d2h_str = date_time_1d2h.strftime("'%Y-%m-%d','%H:%M:%S'")
            index_2h = ','.join([date_time_2h_str, gate_dir])
            index_1d2h = ','.join([date_time_1d2h_str, gate_dir])

            # 特征构造
            # 特征1, 2小时前的特征
            # 特征2, 1天2小时前的特征
            # 存在2小时前的项和1天2小时前的项时, 才做这样的操作:
            if index_2h in save and index_1d2h in save:
                feature1 = ','.join(save[index_2h][3:])
                feature2 = ','.join(save[index_1d2h][3:])
                final_feature = '{},{},{},{}\n'.format(index, feature1, feature2, save[index][-1])
                fw.write(final_feature)


if __name__ == "__main__":
    get_features("key_time_train.csv", "train")
    # get_features("key_time_test.csv", "test")