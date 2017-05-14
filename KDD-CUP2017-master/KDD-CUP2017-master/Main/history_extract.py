# encoding = utf-8

import re
import time
import datetime

def get_key_volume(filename):
    """
    用途: 获取训练集中某个时间段的数据
        4月20日: 目前还不截取某个时间段的数据, 本程序只用来改变时间格式. 获得key_time文件

    实现:
        对于每一行:
            提取出时间
            判断是否在8:00 - 10:00 或 17:00 - 19:00之间
            如果是:
                输出到文件
    """
    begin1 = datetime.datetime.strptime("8:00:00", "%H:%M:%S")
    end1 = datetime.datetime.strptime("10:00:00", "%H:%M:%S")
    begin3 = datetime.datetime.strptime("6:00:00", "%H:%M:%S")
    end3 = datetime.datetime.strptime("8:00:00", "%H:%M:%S")
    begin2 = datetime.datetime.strptime("17:00:00", "%H:%M:%S")
    end2 = datetime.datetime.strptime("19:00:00", "%H:%M:%S")
    begin4 = datetime.datetime.strptime("15:00:00", "%H:%M:%S")
    end4 = datetime.datetime.strptime("17:00:00", "%H:%M:%S")
    with open(filename, 'r') as fr, open("key_time.csv", "w", encoding="utf-8") as fw:
        fr.readline()
        head = "date,time,gate_dir,vehicle_model[0],vehicle_model[1],vehicle_model[2],vehicle_model[3]," \
               "vehicle_model[4],vehicle_model[5],vehicle_model[6],vehicle_model[7]," \
               "has_etc[0],has_etc[1],working_day,total_volume\n"
        fw.write(head)
        for line in fr:
            line = re.split(',', line)
            date_time = line[0].strip('["')
            date_ = datetime.datetime.strptime(re.split(' ', date_time)[0], "%Y-%m-%d")
            time_ = datetime.datetime.strptime(re.split(' ', date_time)[1], "%H:%M:%S")
            gate_dir = line[2].strip('"')
            volume = line[-1].strip('"\n')
            date_str = date_.strftime("%Y-%m-%d")
            time_str = time_.strftime("%H:%M:%S")
            # if begin1 <= time_ < end1 or begin2 <= time_ < end2 or begin3 <= time_ <= end3 or begin4 <= time_ <= end4:
                # fw.write("{},{},{},{}\n".format(date_str, time_str, gate_dir, volume))
            fw.write("'{}','{}',{}".format(date_str, time_str, ",".join(line[2:])))



if __name__ == '__main__':
    # get_key_volume('20min_volume_test_parsed.csv')
    get_key_volume('20min_volume_training_parsed.csv')
