import pandas as pd
import numpy as np


result1 = pd.read_csv("./result/rf_result_0.1658.csv")
result2 = pd.read_csv("./result/rf_result_0.1583.csv")

#finalRes = pd.read_csv("./data/result_combination.csv")

#print((result1["volume"] == finalRes["volume"]))

sumWeight = (1.0-0.1658)+(1.0-0.1583)

result1["volume"] = np.rint(result1["volume"]*(1.0-0.1658)/sumWeight+result2["volume"]*(1.0-0.1583)/sumWeight)

result1.to_csv("./data/result_combination2.csv", index=False)




