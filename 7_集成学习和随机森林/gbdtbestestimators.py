import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    """

    :param regressors: 
    :param X: 
    :param y: 
    :param axes: 
    :param label: 
    :param style: 
    :param data_style: 
    :param data_label: 
    :return: 
    """
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

if __name__ == "__main__":
    """
    gbdt 中弱模型个数不是越多越好，找到最优个数的模型
    """
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.rand(100)

    X_train,X_test,y_train,y_test = train_test_split(X,y)
    gbdt = GradientBoostingRegressor(random_state=42,max_depth=2,n_estimators=120)
    gbdt.fit(X_train,y_train)

    # 得到120次gbdt模型的误差值，staged_predict 在本例中会返回1，2，3...120 个弱模型的值
    errors = [mean_squared_error(y_test,y_pred) for y_pred in gbdt.staged_predict(X_test)]
    # 选误差最小的那个gbdt模型，best_estimators是最优模型的个数
    best_estimators = np.argmin(errors)
    gbdt_best = GradientBoostingRegressor(random_state=42,max_depth=2,n_estimators=best_estimators)
    gbdt_best.fit(X_train,y_train)

    plt.subplot(121)
    # 最优个数点
    min_error = np.min(errors)
    plt.plot([best_estimators, best_estimators], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(best_estimators, min_error, "ko")
    # 画不同个数弱模型组成gbdt 的错误率
    plt.plot(errors,"b.-")

    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.title("Validation error")

    plt.subplot(122)
    plot_predictions([gbdt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("best model (%d tree)"%best_estimators)
    plt.show()




