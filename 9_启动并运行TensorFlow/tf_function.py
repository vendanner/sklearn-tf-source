import tensorflow as tf

def relu(X):
    # with tf.variable_scope("relu",reuse=True):
        # 若没创建threshold，则代码会异常
        # threshold = tf.get_variable("threshold")
    threshold = tf.get_variable("threshold", shape=(),nitializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, threshold, name="relu")

if __name__ == "__main__":
    """
    """
    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    # with tf.variable_scope("relu"):
    #     threshold = tf.get_variable("threshold", shape=(),initializer=tf.constant_initializer(0.0))
    # relus = [relu(X) for i in range(5)]
    relus = []
    for relu_index in range(5):
        # 第一次创建，其后复用
        with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
            relus.append(relu(X))

    output = tf.add_n(relus, name="output")
    # 保存graph
    file_writer = tf.summary.FileWriter("D://tf_logs/relu5", tf.get_default_graph())
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
