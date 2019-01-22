import numpy as  np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    回归树
    """
    np.random.seed(42)
    m  = 200
    X = np.random.rand(m,1)
    y = 4*(X-0.5)**2 + np.random.rand(m,1)/10

    tree_reg1 = DecisionTreeRegressor(random_state=42)
    # 正则化参数
    tree_reg2 = DecisionTreeRegressor(random_state=42,min_samples_leaf=10)
    tree_reg1.fit(X,y)
    tree_reg2.fit(X,y)

    x1 = np.linspace(0, 1, 500).reshape(-1, 1)
    y1 = tree_reg1.predict(x1)
    y2 = tree_reg2.predict(x1)

    plt.subplot(121)
    plt.plot(X, y, "b.")
    plt.plot(x1,y1,'r')
    plt.title("No restrictions", fontsize=14)

    plt.subplot(122)
    plt.plot(X, y, "b.")
    plt.plot(x1, y2, 'r')
    plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)

    plt.show()
    # 很容易看得出来，在没加约束条件min_samples_leaf下，严重过拟合了


