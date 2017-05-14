import pandas as pd
import  numpy as np

def tollGateFeatureExtract():
    routes = pd.read_csv("../data/dataSets/training/routes (table 4).csv")
    links = pd.read_csv("../data/dataSets/training/links (table 3).csv")

    data = pd.DataFrame()
    interSectionNum = {1:0, 2:0, 3:0}
    linkNum = {1:0, 2:0, 3:0}
    for i, id in enumerate([2,3,1,3,1,3]):
        interSectionNum[id] += 1
        lengthSum = 0
        widthSum = 0
        lanesSum = 0
        laneWidthSum = 0
        linkSeq = list(map(int, routes.iloc[i, :]["link_seq"].split(",")))
        print(linkSeq)
        linkNum[id] += len(linkSeq)
        tmpAns = []
        for seq in linkSeq:
            tmpDf = links[links["link_id"] == seq]
            lengthSum += tmpDf["length"].values[0]
            widthSum +=  tmpDf["width"].values[0]
            lanesSum += tmpDf["lanes"].values[0]
            laneWidthSum +=  tmpDf["lane_width"].values[0]
        tmpAns.append([id, lengthSum, widthSum, lanesSum, laneWidthSum])
        record = pd.DataFrame(tmpAns, columns=["tollgate_id", "lengthSum", "widthSum", "lanesSum", "laneWidthSum"])
        data = data.append(record)
    data = data.groupby("tollgate_id").sum()
    data["intersectionNum"] = interSectionNum.values() # 顺序可能有误
    data["linkNum"] = linkNum.values()
    return data



if __name__ == "__main__":
    data = tollGateFeatureExtract()
    print(data)
    data.to_csv("tollgate_feature.csv")