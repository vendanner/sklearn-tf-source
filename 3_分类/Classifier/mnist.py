"""
sklearn 自带的MNIST 数据集(70000张手写识别图)
"""
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

def loadData():
    # fetch_mldata 联网不可用，直接下载到本地加载放在"dataSet/mldata"目录下
    # 网盘地址：https://pan.baidu.com/s/1paV38Ohy_PhZN8th-__txg  or https://raw.githubusercontent.com/amplab/datascience-sp14/master/lab7/mldata/mnist-original.mat
    mnist = fetch_mldata("MNIST original",data_home="../dataSet/")
    return mnist["data"],mnist["target"]

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def analysis(X,index):
    """
    观察数据
    :param X: 
    :return: 
    """
    some_digit = X[index]
    # 数据是28*28的像素图
    some_digit_img = some_digit.reshape(28,28)
    plt.imshow(some_digit_img,cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")
    plt.show()

def plt_precision_recall_vs_threshold(precisions,recalls,thresholds):
    """
    显示 precision,recall，对比图
    :param precisions: 
    :param recalls: 
    :param thresholds: 
    :return: 
    """
    # 注意y轴最后以为没取，为什么？ 因为thresholds 是比precision 少一位
    plt.plot(thresholds,precisions[:-1],"b--",label = "Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label = "Recall")
    plt.xlabel("Threshold")
    plt.legend(loc = "upper left")
    plt.ylim([0,1])

def precision_recall_display(sgd_clf,X_trian,y_train_5,y_train_predict):
    """
    
    :param sgd_clf: 
    :param X_trian: 
    :param y_train_5: 
    :param y_train_predict: 
    :return: 
    """
    # 准确率 = 真正例/（真正例+假正例）；找到是正确的概率
    print("precision_score:",precision_score(y_train_5,y_train_predict))
    # 召回率 = 真正例/（真正例+假反例）；正确的被找到的概率
    print("recall_score:",recall_score(y_train_5,y_train_predict))
    # 准确率和召回率的调和平均,一般用F1 更能体现分类性能(训练集采集样本不同，准确率和召回率会有所波动，但F1一般稳定)
    print("f1_score:",f1_score(y_train_5,y_train_predict))

    # 返回分类的分数值
    y_score = cross_val_predict(sgd_clf,X_trian,y_train_5,cv = 3,method="decision_function")
    # sgd_clf 模型中有个分类阈值的的概念，设定分类阈值大小的不同，分类的结果也许会不同(类似sigmoid 函数的0.5)
    precisions,recalls,thresholds = precision_recall_curve(y_train_5,y_score)
    plt_precision_recall_vs_threshold(precisions,recalls,thresholds)
    plt.show()

def plot_roc_display(fpr,tpr,label=None):
    """
    
    :param fpr: 
    :param tpr: 
    :param label: 
    :return: 
    """
    plt.plot(fpr,tpr,linewidth = 2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

def roc_display(sgd_clf,X_trian,y_train_5,y_train_predict):
    """
    ROC 曲线评估分类模型 真正例率/假正例率
    :param sgd_clf: 
    :param X_trian: 
    :param y_train_5: 
    :param y_train_predict: 
    :return: 
    """
    y_score = cross_val_predict(sgd_clf, X_trian, y_train_5, cv=3, method="decision_function")
    # sgd 默认以0为阈值
    fpr,tpr,thresholds = roc_curve(y_train_5,y_score)
    # plot_roc_display(fpr,tpr)

    # 完美的分类器AUC =1，
    print("sgd auc: ",roc_auc_score(y_train_5,y_score))

    # 对比随机森林模型性能
    forest_clf = RandomForestClassifier()
    y_proba_forest = cross_val_predict(forest_clf,X_trian,y_train_5,cv=3,method="predict_proba")
    forest_fpr,forest_tpr,thresholds = roc_curve(y_train_5,y_proba_forest[:,1])
    plt.plot(fpr,tpr,"b:",label="SGD")
    plot_roc_display(forest_fpr,forest_tpr,"Random Forest")
    plt.legend(loc="bottom right")
    # 随机森林 AUC 比SGD 大，性能优
    print("forest auc: ", roc_auc_score(y_train_5, y_proba_forest[:,1]))

    plt.show()

def twoCategory(X_trian,y_train,X_test,y_test):
    """
    二分类，只区分是否为5
    :param X_trian: 
    :param y_train: 
    :param X_test: 
    :param y_tes: 
    :return: 
    """
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    # random_state变化，SGDClassifier结果也会变
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_trian,y_train_5)

    score = cross_val_score(sgd_clf,X_trian,y_train_5,cv = 3,scoring="accuracy")
    # 正确率在95%，看起来是挺好的；其实是因为样本5只占1/10，这就是样本分布不均匀导致
    # 交叉评估比较适用于回归模型，分类模型用其他指标来衡量
    print("3折交叉评估：\t",score)

    y_train_predict = cross_val_predict(sgd_clf,X_trian,y_train_5,cv = 3)
    conMatrix = confusion_matrix(y_train_5,y_train_predict)
    # 真正例，真反例，假正例，假反例
    print("3折混淆矩阵评估:\t",conMatrix)

    # 准确率、召回率评估
    # precision_recall_display(sgd_clf,X_trian,y_train_5,y_train_predict)

    # ROC 曲线
    roc_display(sgd_clf,X_trian,y_train_5,y_train_predict)


def analysisError(X_trian, y_train, X_test, y_test):
    """
    误差分析
    :param X_trian: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :return: 
    """
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_trian.astype(np.float64))
    sgd_clf = SGDClassifier()
    sgd_clf.fit(X_trian,y_train)
    y_train_pre = cross_val_predict(sgd_clf,X_train_scaler,y_train,cv=3)
    conf_mx = confusion_matrix(y_train,y_train_pre)
    print("多分类 confusion：\n",conf_mx)

    row_sums = conf_mx.sum(axis=1,keepdims=True)
    # 通过观察下图就可以知道那些数字错误率高，那些数字容易被认为是其他数字，我们可以对其进行优化
    norm_conf_mx = conf_mx/row_sums
    np.fill_diagonal(norm_conf_mx,0)
    plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
    plt.show()

def mult_label(X_trian, y_train, X_test, y_test):
    """
    多标签输出
    :param X_trian: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :return: 
    """
    y_train_large = (y_train > 7)
    y_train_odd = (y_train%2 == 1)
    # 2个标签：是否大于7，是否为奇数
    y_multilabel = np.c_[y_train_large,y_train_odd]
    knn_clf = KNeighborsRegressor()
    knn_clf.fit(X_trian,y_multilabel)
    y_train_multilabel_pre = cross_val_predict(knn_clf,X_trian,y_multilabel,cv=3)
    # average="macro" 每个标签在计算f1 时，权重相同
    print("multi_label_f1_score: ",f1_score(y_multilabel,y_train_multilabel_pre,average="macro"))

def multi_output(X_trian, y_train, X_test, y_test):
    """
    多输出分类
    给原始数据图片加噪声，knn 模型还原
    :param X_trian: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :return: 
    """
    noise = np.random.randint(0, 100, (len(X_trian), 784))
    X_trian_mod = X_trian + noise
    y_train_mod = X_trian

    knn_clf = KNeighborsRegressor()
    knn_clf.fit(X_trian_mod,y_train_mod)
    clean_digit = knn_clf.predict([X_trian_mod[100]])
    plot_digit(clean_digit)
    # 输出去噪声后的图片
    plt.show()


if __name__ == "__main__":
    print("begin")
    X ,y = loadData()
    # analysis(X)

    # 打乱数据集原有顺序，分层采样，保证样本分布
    shuffle_index = np.random.permutation(70000)
    X ,y= X[shuffle_index],y[shuffle_index]
    X_trian,y_train,X_test,y_test = X[:60000],y[:60000],X[60000:],y[60000:]

    # twoCategory(X_trian, y_train, X_test, y_test)
    # 误差分析
    # analysisError(X_trian, y_train, X_test, y_test)
    # 多标签
    # mult_label(X_trian, y_train, X_test, y_test)

    multi_output(X_trian, y_train, X_test, y_test)


