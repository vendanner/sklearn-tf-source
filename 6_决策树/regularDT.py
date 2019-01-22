from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap
import numpy as np

if __name__ == "__main__":
    """
    决策树正则化参数对模型的影响
    """
    X,y = make_moons(n_samples=100,noise=0.25,random_state=53)

    # random_state 是参数初始化的随机值
    tree_clf1 = DecisionTreeClassifier(random_state=42)
    tree_clf2 = DecisionTreeClassifier(random_state=42,min_samples_leaf=4)
    tree_clf1.fit(X,y)
    tree_clf2.fit(X,y)

    plt.subplot(121)
    plt.plot(X[:,0][y==0],X[:,1][y==0],"yo")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    x1s = np.linspace(-1.5, 2.5, 100)
    x2s = np.linspace(-1, 1.5, 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = tree_clf1.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    plt.subplot(122)
    plt.plot(X[:,0][y==0],X[:,1][y==0],"yo")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    y_pred1 = tree_clf2.predict(X_new).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred1, alpha=0.3, cmap=custom_cmap)
    plt.show()
    # 增加min_samples_leaf=4 后，训练有较小误差但不会过拟合，这是理想的模型