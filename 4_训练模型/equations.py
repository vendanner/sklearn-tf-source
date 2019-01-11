
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    """
    线性回归正态方程求解
    """
    # 随机生成100个0-1之间的数字
    X = 2 * np.random.rand(100,1)
    # y = 4 + 3 * X 但加入噪声(0-1)
    y = 4 + 3 * X +np.random.rand(100,1)
    # 常量权重为1
    X_b = np.c_[np.ones((100,1)),X]
    # 下面就是线性回归 正态方程公式
    theat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    # y = 4.46963312 + 3.03872799*X
    print("theat:\t",theat)

    # 示意图
    X_new = np.array([[0],[2]])
    X_new_b = np.c_[np.ones((2,1)),X_new]
    y_predict = X_new_b.dot(theat)
    # 看图，可以发现，拟合的还是挺好的；当然你也可以直接用sklearnd的LinearRegression，不用写代码求theat，直接得结果
    plt.plot(X_new,y_predict,"r--")
    plt.plot(X,y,"b.")
    plt.axis([0,2,0,15])
    plt.show()
