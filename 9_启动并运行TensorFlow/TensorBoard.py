"""
可视化训练效果:
    程序运行结束，在cmd 运行tensorboard --logdir=path，然后在浏览器打开http://localhost:6006
"""
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = r"D://tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    scaler = StandardScaler()
    housing = fetch_california_housing()
    scaler_housing_data = scaler.fit_transform(housing.data)
    m,n = scaler_housing_data.shape
    scaler_housing_data_bias = np.c_[np.ones((m,1)),scaler_housing_data]

    # 设置梯度下降超参
    learn_rate = 0.01
    n_epochs = 1000
    # 不指定大小，等到真正训练时再生成
    X = tf.placeholder(shape=(None,n+1),dtype=tf.float32,name='X')
    y = tf.placeholder(shape=(None,1), dtype=tf.float32, name='y')

    # 初始化theta 在-1 - 1 之间
    theta = tf.Variable(tf.random.uniform([n+1,1],-1.0,1.0),name='theta')
    # 计算预测值
    y_pred = tf.matmul(X,theta,name='predictions')

    # 限定变量作用域
    # mse 生成相关操作会被放在loss 框图内
    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")

    # error = y_pred - y
    # 计算损失函数
    # mse = tf.reduce_mean(tf.square(error),name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()
    # 可视化
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    # 设置批量的样本数
    batch_size = 1000
    n_batches = int(np.ceil(m/batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for i in range(n_epochs):
            for batch_index in range (n_batches):
                # 随机批量，没有按顺序
                indices = np.random.randint(m, size=batch_size)
                X_batch = scaler_housing_data_bias[indices]
                y_batch = housing.target.reshape(-1,1)[indices]
                if batch_index%10 == 0:
                    # 每10次保存效果
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = i * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                # 此时才真正确定X，y 大小
                sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        best_theta = theta.eval()
    #  关闭
    file_writer.close()
    print("theta = ",best_theta)