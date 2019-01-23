from sklearn.tree import DecisionTreeRegressor

import numpy as np
import matplotlib.pyplot as plt

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
    x1 = np.linspace(axes[0],axes[1],500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:,0],y,data_style,label = data_label)
    plt.plot(x1,y_pred,style,linewidth = 2,label = label)
    if label or data_label:
        plt.legend(loc="upper center",fontsize=16)
    plt.axis(axes)

if __name__ == "__main__":
    """
    梯度提升树
    """
    np.random.seed(42)
    X = np.random.rand(100,1) - 0.5
    y = 3*X[:,0]**2 + 0.05 * np.random.rand(100)

    tree_reg1= DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg1.fit(X,y)

    # 残差当y
    y2  = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg2.fit(X,y2)

    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg3.fit(X,y3)

    # 上面利用残差训练出了三个模型

    plt.subplot(321)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8],label="$h_1(x_1)$", style="g-", data_label="Training set")

    plt.subplot(322)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$",data_label="Training set")

    plt.subplot(323)
    plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-",
                     data_style="k+",data_label="Residuals")

    plt.subplot(324)
    plot_predictions([tree_reg1,tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

    plt.subplot(325)
    plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-",data_style="k+")

    plt.subplot(326)
    plot_predictions([tree_reg1,tree_reg2,tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8],label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")

    plt.show()

