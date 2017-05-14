from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import  SVR
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from mlxtend.regressor import StackingRegressor


# 建立模型
def buildTrainModel(modelIndex):

    # 输入参数为模型序号，1是GBDT，2是随机森林,3是xgboost, 4是adaboost回归，5是多层感知器，6是k近邻回归 7是模型融合stacking

    linearRf = LinearRegression()
    decesionTreeRf = DecisionTreeRegressor(max_depth=12)

    rf1 = GradientBoostingRegressor(
                                    loss='ls'
                                    , learning_rate=0.05,
                                    n_estimators=150,
                                     subsample=0.7,
                                    # , min_samples_split=2
                                    # , min_samples_leaf=1
                                    max_depth=9
                                    # , init=None
                                    , random_state=0,
                                    max_features=None
                                    # , alpha=0.9
                                    # , verbose=0
                                    # , max_leaf_nodes=None
    )

    rf2 = RandomForestRegressor(n_estimators=130, max_depth=9, max_features=None)

    rf3 = XGBRegressor(objective="reg:linear", n_estimators=100, learning_rate=0.05,
                       max_depth=10, min_child_weight=6, subsample=0.8, colsample_bytree=1.0, silent=True, reg_lambda=1.0, seed=0)


    rf4 = AdaBoostRegressor(base_estimator=rf3, n_estimators=10, loss="linear", learning_rate=0.01)

    rf5 = MLPRegressor(hidden_layer_sizes=10, activation="relu", learning_rate = "adaptive",
                       solver="adam", batch_size="auto", alpha=0.1, early_stopping=True)

    rf6 = KNeighborsRegressor(n_neighbors = 7, weights="uniform", algorithm="kd_tree", p=2)


    stackrf = StackingRegressor(regressors=[rf1, rf2, rf3, rf4],
                               meta_regressor=linearRf)

    rfList = [rf1, rf2, rf3, rf4, rf5, rf6, stackrf]

    return rfList[modelIndex-1]