import pandas as pd
import  numpy as np

def featureCombination():

    feature = pd.read_csv("feature.csv")
    weatherFeature = pd.read_csv("new_weather_train.csv")

    newFeature = pd.merge(feature, weatherFeature)

    newFeature.to_csv("train.csv", index=False)

if __name__ == "__main__":
    featureCombination()