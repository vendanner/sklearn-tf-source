from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(X, y, axes):
    """
    显示原始数据集分布图
    :param X: 
    :param y: 
    :param axes: 
    :return: 
    """
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

def plot_predictions(clf, axes):
    """
    
    :param clf: 
    :param axes: 
    :return: 
    """
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

if __name__ == "__main__":
    poly_kernel_svm_clf3 = Pipeline((
        ("standard",StandardScaler()),
        # 多项式三次方，
        ("poly_svc",SVC(kernel='poly',degree=3,coef0=1,C=5))
    ))
    poly_kernel_svm_clf10 = Pipeline((
        ("standard",StandardScaler()),
        # 多项式10次方，coef0代表多项式之间的关系
        ("poly_svc",SVC(kernel='poly',degree=10,coef0=100,C=5))
    ))

    X,y = datasets.make_moons(n_samples=100,noise=0.15,random_state=42)
    poly_kernel_svm_clf3.fit(X,y)
    poly_kernel_svm_clf10.fit(X,y)

    plt.subplot(121)
    plot_predictions(poly_kernel_svm_clf3,[-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

    plt.subplot(122)
    plot_predictions(poly_kernel_svm_clf10,[-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

    plt.show()