# encoding = utf-8

'''
参数: 开始日期(精确到日)
    : 天数
    : 时间段T1, 时间段T2

对于每一天:
    begintime = [T1.begintime, T1.endtime]
    endtime = [T1.begintime, T1.endtime]

    遍历整个数据:
        如果该条数据在begintime 和 endtime中间
            加入save数组[gate_dir].

    对数组的每个gate_dir进行EMA, 直接填充在当前数组后面

    提取出save数组的后半段res

    把begintime和endtime都加两个小时:
    从begintime开始, 每20分钟输出一次res的值
'''


def EMA(arr):
    newlen = len(arr)
    # alpha = 2 / (newlen + 1)
    alpha = 0.5
    arrpre = [arr[0]] * newlen
    for index in range(newlen-1):
        index += 1
        arrpre[index] = arrpre[index-1] + alpha * (arr[index-1] - arrpre[index-1])
    return arrpre

ndays = 1 # 天数
arr = [617, 630, 648, 739, 740, 740, 740, 740, 740, 740, 740, 740]
print(EMA(arr))