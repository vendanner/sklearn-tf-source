"""
tensorflow 构建DNN 网络
"""
from datetime import datetime
import tensorflow as tf
from sklearn.datasets import fetch_mldata
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data



if __name__ == "__main__":
    """
    """
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = r"D://tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    mnist = input_data.read_data_sets("/tmp/data/")
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
    # 这里rank 不为(None，1)，因为tensorflow logits输出是一维
    y = tf.placeholder(tf.int64,shape=(None),name="y")
    # 定义全连接的神经网络
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name='hidden1')
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name='outputs')
    # 定义损失函数 - 交叉熵
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name="loss")
    # 训练模型，loss 最小
    learn_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        train_op = optimizer.minimize(loss)

    # 评估模型
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits,y,1)
        accurary =tf.reduce_mean(tf.cast(correct,tf.float32))

    # 可视化
    train_summary = tf.summary.scalar('accurary', accurary)
    merged = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # 执行
    n_epochs = 1
    batch_size = 50
    with tf.Session() as sess:
        init.run()

        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples//batch_size):
                X_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(train_op,feed_dict={X:X_batch,y:y_batch})
                summary,acc_train = sess.run([merged,accurary],feed_dict={X:X_batch,y:y_batch})
                # acc_train = accurary.eval(feed_dict={X:X_batch,y:y_batch})
                acc_test = accurary.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
                file_writer.add_summary(summary, epoch * batch_size + iteration)
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


