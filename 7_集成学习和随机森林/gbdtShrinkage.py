import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


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
    gbdt 在学习率小的情况下需要更多的模型集成才能得到好效果
    看输出图形可知，
    learning_rate 类似梯度下降的步长，它是每个模型可以拟合数值大小，learning_rate大单个模型拟合程度高，learning_rate小需要越多的模型
    n_estimators 是模型数量，n_estimators越大gbdt 拟合越好，但这个数值也有个极限值，超过了模型效果也不会变好
    """
    np.random.seed(42)
    X = np.random.rand(100,1) - 0.5
    y = 3*X[:,0]**2 + 0.05 * np.random.rand(100)

    # X_train,X_test,y_train,y_test = train_test_split(X,y)
    # 训练4个GBRT 模型，学习率和模型个数不同
    gbrt_reg1 = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1, random_state=42)
    gbrt_reg2 = GradientBoostingRegressor(max_depth=2, n_estimators=50, learning_rate=0.1, random_state=42)
    gbrt_reg3 = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1, random_state=42)
    gbrt_reg4 = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=1, random_state=42)
    gbrt_reg1.fit(X,y)
    gbrt_reg2.fit(X, y)
    gbrt_reg3.fit(X, y)
    gbrt_reg4.fit(X, y)

    plt.subplot(221)
    plot_predictions([gbrt_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("learning_rate=0.1,n_estimators=3")

    plt.subplot(222)
    plot_predictions([gbrt_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate=0.1,n_estimators=50")

    plt.subplot(223)
    plot_predictions([gbrt_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate=1,n_estimators=3")

    plt.subplot(224)
    plot_predictions([gbrt_reg4], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate=1,n_estimators=200")
    plt.show()


