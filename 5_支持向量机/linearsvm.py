import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    # important
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

if __name__ == "__main__":
    data = datasets.load_iris()
    X = data["data"][:,(2,3)]
    y = (data["target"] == 2).astype(np.float64)

    scaler = StandardScaler()
    svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
    svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)
    scaler_svm_clf1 = Pipeline((
        ("standardscaler",scaler),
        # 设定超参C 和损失函数
        ("linear_svc",svm_clf1),
    ))
    scaler_svm_clf2 = Pipeline((
        ("standardscaler",scaler),
        # 设定超参C 和损失函数
        ("linear_svc",svm_clf2),
    ))


    # 训练模型
    scaler_svm_clf1.fit(X,y)
    scaler_svm_clf2.fit(X, y)
    # 抽取模型训练结果

    # 恢复原始值
    b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
    b2 = svm_clf2.decision_function([-scaler.mean_ /scaler.scale_])
    w1 = svm_clf1.coef_[0]/scaler.scale_
    w2 = svm_clf2.coef_[0]/scaler.scale_
    svm_clf1.intercept_ = (np.array([b1]))
    svm_clf2.intercept_ = (np.array([b2]))
    svm_clf1.coef_ = np.array([w1])
    svm_clf2.coef_ = np.array([w2])
    # 提取支持向量；linearSVC 不自动保存，SVC 才保存
    t = y * 2 - 1    # 不懂为什么要如此变换，理论上y 既可
    # 根据感知机理论，分类正确是y * g(x) > 1
    support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
    support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
    svm_clf1.support_vectors_ = X[support_vectors_idx1]
    svm_clf2.support_vectors_ = X[support_vectors_idx2]
    # 可以画出svm 的分割线

    # svm 直接出分类结果
    print(svm_clf1.predict([[5.5,1.7]]))
    # 第一幅图
    # subplot 121 整个窗口分为1行2列，当前是第一幅图
    plt.subplot(121)
    plt.plot(X[:,0][y==1],X[:,1][y==1],'g^',label="Iris-Virginica")
    plt.plot(X[:,0][y==0],X[:,1][y==0],'bs',label="Iris-Versicolor")
    print(svm_clf1)
    plot_svc_decision_boundary(svm_clf1, 4, 6)
    plt.axis([4, 6, 0.8, 2.8])
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
    plt.legend(loc="upper left", fontsize=14)

    # 第二幅图
    plt.subplot(122)
    plt.plot(X[:,0][y==1],X[:,1][y==1],'g^',label="Iris-Virginica")
    plt.plot(X[:,0][y==0],X[:,1][y==0],'bs',label="Iris-Versicolor")
    plot_svc_decision_boundary(svm_clf2, 4, 6)
    plt.axis([4, 6, 0.8, 2.8])
    plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
    plt.show()


