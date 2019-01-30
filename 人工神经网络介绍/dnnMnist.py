"""
tensorflow estimator 直接训练mnist识别器
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def loadData():
    # fetch_mldata 联网不可用，直接下载到本地加载放在"dataSet/mldata"目录下
    # 网盘地址：https://pan.baidu.com/s/1paV38Ohy_PhZN8th-__txg  or https://raw.githubusercontent.com/amplab/datascience-sp14/master/lab7/mldata/mnist-original.mat
    mnist = fetch_mldata("MNIST original",data_home="dataSet/")
    return mnist["data"],mnist["target"]

if __name__ == "__main__":
    """
    
    """
    X,y = loadData()
    standard = StandardScaler()
    X_standard = standard.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X_standard,y,random_state=42,test_size=0.3)
    # 注意标签是要int 类型
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    feature_cols =[tf.feature_column.numeric_column("X", shape=[28 * 28])]

    # 300*100 神经元，10分类
    dnn_clf = tf.estimator.DNNClassifier(feature_columns=feature_cols,hidden_units=[300,100],n_classes=10)
    input_fn = tf.estimator.inputs.numpy_input_fn(x={"X":X_train},y=y_train,num_epochs=40,batch_size=1000,shuffle=True)
    dnn_clf.train(input_fn = input_fn)

    test_fn = tf.estimator.inputs.numpy_input_fn(x={"X":X_test},y=y_test,shuffle=False)
    score = dnn_clf.evaluate(input_fn=test_fn)
    # 成功率0.96 效率遥遥领先
    print("score\n",score)



