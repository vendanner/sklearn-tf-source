from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

import os

PROJECT_ROOT_DIR = '.'

def image_path(file_id):
    """
    
    :param file_id: 
    :return: 
    """
    return os.path.join(PROJECT_ROOT_DIR,"output",file_id)

if __name__ == "__main__":
    """
    决策树分类
    """
    iris = load_iris()
    X = iris["data"][:,2:]
    y = iris["target"]
    # 决策树预剪枝 设定为2
    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X,y)
    # export_graphviz(
    #     tree_clf,
    #     out_file=image_path("iris_tree.dot"),
    #     feature_names=iris.feature_names[2:],
    #     class_names=iris.target_names,
    #     rounded=True,
    #     filled=True
    # )

    # 展示三个类别的数据集
    plt.plot(X[:,0][y==0],X[:,1][y==0],"yo",label="Iris-Setosa")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris-Virginca")

    # 展示特征值分割点,数值从iris_tree.dot 观察
    plt.plot([2.5,2.5],[0,3],'k-')
    plt.plot([2.5, 7.5], [1.75, 1.75], 'k--')

    plt.text(1.40, 1.0, "Depth=0", fontsize=15)
    plt.text(3.2, 1.80, "Depth=1", fontsize=13)
    plt.axis([0,7.5,0,3])
    plt.legend()
    plt.show()
    # 图中观察，Iris-Setosa 分类完全正确无错误点，而Iris-Versicolor、Iris-Virginca 有分类错误点