import pandas as pd
from datetime import datetime, timedelta

def cal_mean():
    data = pd.read_csv("../data/training_20min_avg_volume_original.csv")
    data["start_time"] = data["time_window"].apply(lambda x: datetime.strptime(x.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S"))
    start_time_tmp = datetime(2016, 9, 19, 0, 0, 0)
    start_time_list = []
    # 72 time window per day; 29 days in trainset
    for i in range(72*29):
        start_time_list.append(start_time_tmp + timedelta(minutes=20*i))

    # fill
    tollgate = [1, 2, 3]
    direction = [0, 1]
    for i in tollgate:
        for j in direction:
            df_list = []
            for k in start_time_list:
                if (i != 2 or j != 1) and data[data.start_time == k][data.tollgate_id == i][data.direction == j].empty:
                    time_window_tmp = "[" + k.strftime("%Y-%m-%d %H:%M:%S") + "," + (k + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S") + ")"
                    df_list.append([i, time_window_tmp, j, 0, k])
            df = pd.DataFrame(df_list, columns=data.columns)
            data = data.append(df).reset_index(drop=True)

    data.to_csv("../data/training_20min_avg_volume_fill.csv")

    #print(data.info())
    data["date"] = data["start_time"].apply(lambda x: datetime.date(x))
    data["time"] = data["start_time"].apply(lambda x: datetime.time(x))
    data["day_of_week"] = data["start_time"].apply(lambda x: x.isoweekday())
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x>=6 else 0)
    data = data.sort(["date", "time"])


    # drop 10-1 7 days holiday
    drop_date_start = datetime.strptime('2016-09-30', '%Y-%m-%d').date()
    data_drop = data.copy()
    for i in range(9):
        data_drop = data_drop[data_drop["date"] != drop_date_start + timedelta(hours = i*24)]



    #predict format generation
    match_time = datetime.strptime('08:00:00', '%H:%M:%S')
    match_time_list = [(match_time + timedelta(minutes = i * 20)).time() for i in range(6)]
    match_time = datetime.strptime('17:00:00', '%H:%M:%S')
    match_time_list += [(match_time + timedelta(minutes=i * 20)).time() for i in range(6)]

    predict_start_time = datetime(2016, 10, 18, 0, 0, 0)
    predict_time = [predict_start_time + timedelta(minutes=i // 5 * 20) for i in range(5 * 72 * 7)]
    predict_gate = [[1,1,2,3,3] for i in range(72 * 7)]
    predict_gate = [item for sublist in predict_gate for item in sublist]
    predict_direction = [[0,1,0,0,1] for i in range(72 * 7)]
    predict_direction = [item for sublist in predict_direction for item in sublist]

    predict_df = pd.DataFrame()
    predict_df["tollgate_id"] = predict_gate
    predict_df["time_window"] = pd.DataFrame(predict_time,  columns=["date_time"])["date_time"].apply(lambda x: "[" + x.strftime("%Y-%m-%d %H:%M:%S") + "," + (x + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S") + ")")
    predict_df["direction"] = predict_direction
    predict_df["date_time"] = predict_time
    predict_df = predict_df[predict_df["date_time"].apply(lambda x: x.time()).isin(match_time_list)]


    predict_df["date"] = predict_df["date_time"].apply(lambda x: datetime.date(x))
    predict_df["time"] = predict_df["date_time"].apply(lambda x: datetime.time(x))
    predict_df["day_of_week"] = predict_df["date_time"].apply(lambda x: x.isoweekday())
    predict_df["is_weekend"] = predict_df["day_of_week"].apply(lambda x: 1 if x>=6 else 0)

    #predict
    data_rush = data_drop.copy()
    data_rush = data_rush[data_rush["time"].isin(match_time_list)]
    volume_tmp = data_rush.groupby(["time", "is_weekend","tollgate_id", "direction"], as_index=False)["volume"].mean()
    prediction = pd.merge(predict_df, volume_tmp, on = ["time", "is_weekend" ,"tollgate_id", "direction"], how = "left")
    #print(prediction.info())
    #prediction["volume"] = prediction["volume"].apply(lambda x: math.floor(x))
    #prediction["volume"] = prediction["volume"]
    #print(prediction.info())
    prediction[["tollgate_id", "time_window", "direction", "volume"]].to_csv("prediction.csv", index = False)

if __name__ == "__main__":
    cal_mean()
