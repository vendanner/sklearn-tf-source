"""
线性回归梯度下降
"""
import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    scaler = StandardScaler()
    housing = fetch_california_housing()
    scaler_housing_data = scaler.fit_transform(housing.data)
    m,n = scaler_housing_data.shape
    scaler_housing_data_bias = np.c_[np.ones((m,1)),scaler_housing_data]

    # 设置梯度下降超参
    learn_rate = 0.01
    n_epochs = 1000
    # 创建tf 节点
    X = tf.Variable(scaler_housing_data_bias,dtype=tf.float32,name='X')
    y = tf.Variable(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
    # 初始化theta 在-1 - 1 之间
    theta = tf.Variable(tf.random.uniform([n+1,1],-1.0,1.0),name='theta')
    # 计算预测值
    y_pred = tf.matmul(X,theta,name='predictions')
    error = y_pred - y
    # 计算损失函数
    mse = tf.reduce_mean(tf.square(error),name='mse')
    # # 下面2行代码合起来就是梯度下降 w更新
    # # gradients = 2/m *tf.matmul(X,error,transpose_a=True)
    # # 直接调用gradients函数求梯度
    # gradients = tf.gradients(mse, [theta])[0]
    # # assign 将theta - learn_rate*gradients 赋予theta
    # training_op = tf.assign(theta,theta - learn_rate*gradients)
    # init = tf.global_variables_initializer()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for i in range(n_epochs):
    #         if i%100 == 0:
    #             print("Epoch",i," MSE = ",mse.eval())
    #         sess.run(training_op)
    #     best_theta = theta.eval()

    # 介绍更简单的方法，tf集成的优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(n_epochs):
            if i%100 == 0:
                print("Epoch",i," MSE = ",mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        save_path = saver.save(sess,"tmp/best_theta.ckpt")
    print("theta = ",best_theta)