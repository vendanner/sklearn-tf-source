import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    """
    softmax 回归
    """
    iris = datasets.load_iris()
    X = iris["data"][:,(2,3)]
    y = iris["target"]
    # LogisticRegression 原本输出多类别的概率，"multinomial" 改为softmax只输出分类结果
    softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10)
    softmax_reg.fit(X,y)

    x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(),x1.ravel()]
    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)
    # 不同种类不同形状标识
    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")


    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    # 填充点 = 背景着色
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    # 等高线
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])

    plt.legend()
    plt.show()