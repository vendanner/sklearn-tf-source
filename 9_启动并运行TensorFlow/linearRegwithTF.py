import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    """
    tensorflow 线性回归
    """
    housing = fetch_california_housing()
    # m个n维数据
    m,n = housing.data.shape
    print(m,n)
    # 增加偏置数值固定为1
    housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]

    X = tf.constant(housing_data_plus_bias,dtype=tf.float32,name = "X")
    y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)) ,XT),y)

    with tf.Session() as sess :
        theta_value = theta.eval()

    # 输出权重w0 - w8
    print(theta_value)

