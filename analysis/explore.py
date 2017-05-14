import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime,timedelta

data = pd.read_csv("../data/training_20min_avg_volume_original.csv")

data["start_time"] = data["time_window"].apply(lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))
data["date"] = data["start_time"].apply(lambda x: datetime.date(x))
data["time"] = data["start_time"].apply(lambda x: datetime.time(x))
data["day_of_week"] = data["start_time"].apply(lambda x: x.isoweekday())
data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x>=6 else 0)
data = data.sort(["date", "time"])
drop_date_start = datetime.strptime('2016-09-30', '%Y-%m-%d').date()
data_drop = data
for i in range(9):
    data = data_drop[data_drop["date"] != drop_date_start + timedelta(hours = i*24)]
print (data.columns)
#data.to_csv("drop_national_day_training.csv", index=False)



id1 = data[data["tollgate_id"] == 1]
id1out = id1[id1["direction"] == 1]
id1in = id1[id1["direction"] == 0]
id2 = data[data["tollgate_id"] == 2]
id2in = id2[id2["direction"] == 0]
id3 = data[data["tollgate_id"] == 3]
id3out = id3[id3["direction"] == 1]
id3in = id3[id3["direction"] == 0]
print(id1in["volume"].values.shape)
plt.plot(id2in["volume"].values)
plt.show()

'''
id1outMean = id1out["volume"].mean()
id1inMean = id1in["volume"].mean()
print(id1inMean, id1outMean)
id2inMean = id2in["volume"].mean()
print(id2inMean)
id3outMean = id3out["volume"].mean()
id3inMean = id3in["volume"].mean()
print(id3inMean, id3outMean)

resultDF = pd.DataFrame()
start_time_window1 = datetime(2016, 10, 18, 8, 0, 0)
start_time_window2 = datetime(2016, 10, 18, 17, 0, 0)

fw = open("ave_result_dropNationalDay.csv", 'w')
fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
for k in [1, 2, 3]:
    for dir in [0, 1]:
        for i in range(7):
            start_time_window = start_time_window1 + timedelta(days=1 * i)
            for j in range(6):
                        hour = start_time_window.hour
                        minute = start_time_window.minute
                        print(hour, minute)
                        volumeAns = 0
                        if k == 2 and dir == 1:
                            continue
                        if k == 2 and dir == 0:
                            volumeAns = id2in[(id2in["time_window_hour"] == hour) & (id2in["time_window_minute"] == minute)]["volume"].mean()
                            #print(volumeAns)
                        if k == 1 and dir == 0:
                            volumeAns = id1in[(id1in["time_window_hour"] == hour) & (id1in["time_window_minute"] == minute)]["volume"].mean()
                        if k == 1 and dir == 1:
                            volumeAns = id1out[(id1out["time_window_hour"] == hour) & (id1out["time_window_minute"] == minute)]["volume"].mean()
                        if k == 3 and dir == 0:
                            volumeAns = id3in[(id3in["time_window_hour"] == hour) & (id3in["time_window_minute"] == minute)]["volume"].mean()
                        if k == 3 and dir == 1:
                            volumeAns =  id3out[(id3out["time_window_hour"] == hour) & (id3out["time_window_minute"] == minute)]["volume"].mean()
                        time_window_end = start_time_window + timedelta(minutes=20)
                        #print (start_time_window, time_window_end)
                        out_line = ','.join(['"' + str(k) + '"',
                                             '"[' + str(start_time_window) + ',' + str(time_window_end) + ')"',
                                             '"' + str(dir) + '"',
                                             '"' + str(volumeAns) + '"',
                                             ]) + '\n'
                        start_time_window = time_window_end
                        fw.writelines(out_line)
fw.close()


fw = open("ave_result.csv", 'a')
for k in [1, 2, 3]:
    for dir in [0, 1]:
        for i in range(7):
            start_time_window = start_time_window2 + timedelta(days=1 * i)
            for j in range(6):
                        hour = start_time_window.hour
                        minute = start_time_window.minute
                        volumeAns = 0
                        if k == 2 and dir == 1:
                            continue
                        if k == 2 and dir == 0:
                            volumeAns = id2in[(id2in["time_window_hour"] == hour) & (id2in["time_window_minute"] == minute)]["volume"].mean()
                            #print(volumeAns)
                        if k == 1 and dir == 0:
                            volumeAns = id1in[(id1in["time_window_hour"] == hour) & (id1in["time_window_minute"] == minute)]["volume"].mean()
                        if k == 1 and dir == 1:
                            volumeAns = id1out[(id1out["time_window_hour"] == hour) & (id1out["time_window_minute"] == minute)]["volume"].mean()
                        if k == 3 and dir == 0:
                            volumeAns = id3in[(id3in["time_window_hour"] == hour) & (id3in["time_window_minute"] == minute)]["volume"].mean()
                        if k == 3 and dir == 1:
                            volumeAns =  id3out[(id3out["time_window_hour"] == hour) & (id3out["time_window_minute"] == minute)]["volume"].mean()
                        time_window_end = start_time_window + timedelta(minutes=20)
                        #print (start_time_window, time_window_end)
                        out_line = ','.join(['"' + str(k) + '"',
                                             '"[' + str(start_time_window) + ',' + str(time_window_end) + ')"',
                                             '"' + str(dir) + '"',
                                             '"' + str(volumeAns) + '"',
                                             ]) + '\n'
                        start_time_window = time_window_end
                        fw.writelines(out_line)
fw.close()

'''








