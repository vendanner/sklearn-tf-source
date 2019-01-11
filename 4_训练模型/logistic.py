import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    逻辑回归测试
    """
    iris = datasets.load_iris()
    X = iris["data"][:,3:]
    # 转化为2分类问题
    y = (iris["target"] == 2).astype(np.int)
    log_reg = LogisticRegression()
    log_reg.fit(X,y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    # 输出2个类别的概率
    y_predict_proba = log_reg.predict_proba(X_new)

    plt.plot(X_new,y_predict_proba[:,1],'g-',label="Iris-Virginica")
    plt.plot(X_new,y_predict_proba[:,0],'b--',label="Not Iris-Virginica")
    plt.xlabel("petal width")
    plt.ylabel("prob")
    plt.legend()
    # 你会发现花瓣大的就是Iris-Virginica
    # 但是在1.6 附近2者的概率都差不多在0.5，容易误判 - svm 就是解决这类问题
    plt.show()