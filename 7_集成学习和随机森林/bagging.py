from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

if __name__ == "__main__":
    """
    bagging
    在sklearn 中想复现，random_state 一定要设置相同
    """
    X,y = make_moons(n_samples=500,noise=0.25,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

    # 500 个决策树，自举法抽样100个样本
    bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42),max_samples=100,n_estimators=500,bootstrap=True,n_jobs=-1,oob_score=True)
    dt_clf = DecisionTreeClassifier(random_state=42)
    bag_clf.fit(X_train,y_train)
    dt_clf.fit(X_train,y_train)

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    x1s = np.linspace(-1.5,2.5,100)
    x2s = np.linspace(-1,1.5,100)
    x1,x2 = np.meshgrid(x1s,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_bag_pred = bag_clf.predict(X_new).reshape(x1.shape)
    y_dt_pred = dt_clf.predict(X_new).reshape(x1.shape)
    print(y_bag_pred.shape)
    plt.subplot(121)
    plt.plot(X[:,0][y==0],X[:,1][y==0],"yo")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.contourf(x1, x2, y_dt_pred, alpha=0.3, cmap=custom_cmap)
    plt.title("DecisionTreeClassifier")
    plt.axis([-1.5, 2.5, -1, 1.5])

    plt.subplot(122)
    plt.plot(X[:,0][y==0],X[:,1][y==0],"yo")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.contourf(x1, x2, y_bag_pred, alpha=0.3, cmap=custom_cmap)
    plt.title("BaggingClassifier")
    plt.axis([-1.5, 2.5, -1, 1.5])

    """
    bagging_oob_score 0.9333333333333333
    bagging_test_score 0.952
    看到2者很相近，在工程中我们那oob 来评估模型 - 尽可能不碰测试数据
    """
    print("bagging_oob_score",bag_clf.oob_score_)
    print("bagging_test_score", accuracy_score(y_test,bag_clf.predict(X_test)))
    plt.show()
